# SPDX-License-Identifier: Apache-2.0
"""The plugin's hooks into vLLM 0.32. Each one is a no-op for every model without mammoth's modeling_overrides."""
from __future__ import annotations

import contextvars
import functools
import importlib
import logging
import os
from typing import Any

from . import qev
from .tags import merge_ranges

logger = logging.getLogger("vllm.mammoth")

# The scheduled batch whose attention metadata is being built (set around prepare_attn / _build_attention_metadata).
_BATCH_REQ_IDS: contextvars.ContextVar[Any] = contextvars.ContextVar("mammoth_batch_req_ids", default=None)


#: Set by the config hook in the process that builds the model config; inherited by the engine/worker processes.
MODEL_ENV = "VLLM_MAMMOTH_MODEL"


def _is_mammoth(model_config) -> bool:
    try:
        return qev.mammoth_overrides(getattr(model_config, "hf_config", None)) is not None
    except ValueError:
        return False


def _serving_mammoth() -> bool:
    """Whether this process serves mammoth: the current vLLM config if one is set, else the inherited flag."""
    try:
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
    except Exception:  # noqa: BLE001
        cfg = None
    if cfg is not None and getattr(cfg, "model_config", None) is not None:
        return _is_mammoth(cfg.model_config)
    return os.environ.get(MODEL_ENV) == "1"


# 1. Config: the block-bidirectional checkpoint uses vLLM's per-request bidirectional ranges ("mm_prefix"), so the
#    attention layers and backends set them up (TRITON_ATTN / FLEX_ATTENTION / FlashAttention 4 support them).
def patch_model_config() -> None:
    import dataclasses

    from vllm.config.model import ModelConfig

    original = ModelConfig.get_model_arch_config
    if getattr(original, "_mammoth", False):
        return

    @functools.wraps(original)
    def get_model_arch_config(self):
        arch = original(self)
        if _is_mammoth(self):
            os.environ[MODEL_ENV] = "1"
            if not arch.is_mm_prefix_lm:
                arch = dataclasses.replace(arch, is_mm_prefix_lm=True)
        return arch

    get_model_arch_config._mammoth = True  # type: ignore[attr-defined]
    ModelConfig.get_model_arch_config = get_model_arch_config  # type: ignore[method-assign]


# 1b. VLLM_BATCH_INVARIANT=1: vLLM 0.32 refuses to start any GatedDeltaNet model (Qwen3.5 family) in batch-invariant
#     mode because GDNAttentionBackend does not declare support. A mammoth read is prefill only; the Triton GDN prefill
#     kernels (causal conv1d + chunked delta rule) run per sequence over cu_seqlens, and batch-invariant mode already
#     swaps the matmuls/norms. For mammoth only, the plugin declares support (opt out:
#     VLLM_MAMMOTH_GDN_BATCH_INVARIANT=0); measured bit-identical alone vs. inside concurrent batches (prefill reads).
def patch_gdn_batch_invariance() -> bool:
    if os.environ.get("VLLM_MAMMOTH_GDN_BATCH_INVARIANT", "1") == "0":
        return False
    try:
        mod = importlib.import_module("vllm.v1.attention.backends.gdn_attn")
    except Exception as e:  # noqa: BLE001
        logger.debug("mammoth: gdn_attn not importable: %s", e)
        return False
    cls = getattr(mod, "GDNAttentionBackend", None)
    if cls is None:
        return False
    original = cls.__dict__.get("supports_batch_invariance")
    if getattr(getattr(original, "__func__", None), "_mammoth", False):
        return True
    fallback = cls.supports_batch_invariance

    def supports_batch_invariance(c) -> bool:
        return True if _serving_mammoth() else fallback()

    supports_batch_invariance._mammoth = True  # type: ignore[attr-defined]
    cls.supports_batch_invariance = classmethod(supports_batch_invariance)  # type: ignore[method-assign]
    _force_triton_gdn_prefill()
    return True


def _force_triton_gdn_prefill() -> None:
    """Under VLLM_BATCH_INVARIANT=1 use the Triton/FLA GDN prefill kernels: measured on an RTX PRO 6000 (26 Oct
    2026), the FlashInfer GDN prefill kernel that `auto` picks changes results with the batch (up to 0.014 in
    probability), the Triton kernels are bit-identical alone and inside concurrent batches. Same as passing
    `--gdn-prefill-backend triton`."""
    import vllm.envs as envs

    try:
        mod = importlib.import_module("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    except Exception as e:  # noqa: BLE001
        logger.debug("mammoth: qwen_gdn_linear_attn not importable: %s", e)
        return
    resolve = getattr(mod, "_resolve_gdn_prefill_backend", None)
    if resolve is None or getattr(resolve, "_mammoth", False):
        return

    @functools.wraps(resolve)
    def _resolve_gdn_prefill_backend(*args, **kwargs):
        requested, chosen = resolve(*args, **kwargs)
        cfg = args[0] if args else kwargs.get("vllm_config")
        mammoth = _is_mammoth(getattr(cfg, "model_config", None)) if cfg is not None else _serving_mammoth()
        if envs.VLLM_BATCH_INVARIANT and mammoth and chosen != "triton":
            if not _resolve_gdn_prefill_backend._logged:
                logger.info("vllm-mammoth: GDN prefill backend %s -> triton (batch-invariant)", chosen)
                _resolve_gdn_prefill_backend._logged = True
            return requested, "triton"
        return requested, chosen

    _resolve_gdn_prefill_backend._mammoth = True  # type: ignore[attr-defined]
    _resolve_gdn_prefill_backend._logged = False  # type: ignore[attr-defined]
    mod._resolve_gdn_prefill_backend = _resolve_gdn_prefill_backend


# 2. Worker: give tagged requests their two ranges when the attention metadata is built.
def _with_batch(method, req_ids_of):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        mammoth = getattr(self, "_mammoth_serving", None)
        if mammoth is None:
            mammoth = _is_mammoth(getattr(self, "model_config", None))
            try:
                self._mammoth_serving = mammoth
            except Exception:  # noqa: BLE001
                pass
        if not mammoth:
            return method(self, *args, **kwargs)
        token = _BATCH_REQ_IDS.set(req_ids_of(self, args, kwargs))
        try:
            return method(self, *args, **kwargs)
        finally:
            _BATCH_REQ_IDS.reset(token)
    wrapper._mammoth = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_build_attn_metadata(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        req_ids = _BATCH_REQ_IDS.get()
        if req_ids and not kwargs.get("for_cudagraph_capture", False):
            kwargs["mm_req_doc_ranges"] = merge_ranges(kwargs.get("mm_req_doc_ranges"), req_ids)
        return fn(*args, **kwargs)
    wrapper._mammoth = True  # type: ignore[attr-defined]
    return wrapper


def patch_runner_v2() -> bool:
    """Model Runner V2 (the default in 0.32): DefaultModelState / MambaHybridModelState.prepare_attn call
    attn_utils.build_attn_metadata; Qwen3.5 (hybrid) uses MambaHybridModelState."""
    patched = False
    for modname, clsname in (("vllm.v1.worker.gpu.model_states.default", "DefaultModelState"),
                             ("vllm.v1.worker.gpu.model_states.mamba_hybrid", "MambaHybridModelState")):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            logger.debug("mammoth: %s not importable: %s", modname, e)
            continue
        fn = getattr(mod, "build_attn_metadata", None)
        cls = getattr(mod, clsname, None)
        meth = cls.__dict__.get("prepare_attn") if cls is not None else None
        if fn is None or meth is None:
            logger.warning("vllm-mammoth: %s.%s/build_attn_metadata not found; this vLLM is not supported", modname,
                           clsname)
            continue
        if not getattr(fn, "_mammoth", False):
            mod.build_attn_metadata = _wrap_build_attn_metadata(fn)
        if not getattr(meth, "_mammoth", False):
            def req_ids_of(self, args, kwargs):
                batch = kwargs.get("input_batch", args[0] if args else None)
                return list(getattr(batch, "req_ids", None) or [])
            cls.prepare_attn = _with_batch(meth, req_ids_of)
        patched = clsname == "MambaHybridModelState" or patched
    return patched


def patch_runner_v1() -> bool:
    """Model Runner V1: _build_attention_metadata builds one CommonAttentionMetadata with mm_req_doc_ranges."""
    try:
        mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    except Exception as e:  # noqa: BLE001
        logger.debug("mammoth: V1 runner not importable: %s", e)
        return False
    runner = getattr(mod, "GPUModelRunner", None)
    cam = getattr(mod, "CommonAttentionMetadata", None)
    meth = runner.__dict__.get("_build_attention_metadata") if runner is not None else None
    if runner is None or cam is None or meth is None:
        return False
    if not getattr(cam, "_mammoth", False):
        class MammothCommonAttentionMetadata(cam):  # type: ignore[misc,valid-type]
            _mammoth = True

            def __init__(self, *args, **kwargs):
                req_ids = _BATCH_REQ_IDS.get()
                if req_ids and "mm_req_doc_ranges" in kwargs:
                    kwargs["mm_req_doc_ranges"] = merge_ranges(kwargs["mm_req_doc_ranges"], req_ids)
                super().__init__(*args, **kwargs)

        MammothCommonAttentionMetadata.__name__ = cam.__name__
        MammothCommonAttentionMetadata.__qualname__ = cam.__qualname__
        mod.CommonAttentionMetadata = MammothCommonAttentionMetadata
    if not getattr(meth, "_mammoth", False):
        def req_ids_of(self, args, kwargs):
            return list(self.input_batch.req_ids)
        runner._build_attention_metadata = _with_batch(meth, req_ids_of)
    return True


# 3. API server: /v1/systemone uses MammothStrategy for such checkpoints and keeps the request's own JSON values
#    (qev renders non-string instructions/criteria as indented JSON; vLLM's parser would str() them).
_ACTIVE = {"strategy": False}


def _activate_question_rules() -> None:
    from .compat import question_types, serving
    from .strategy import MAX_OPTIONS

    question_types.ChoiceQuestion.label_set = tuple(qev.SLOT_LABELS[:MAX_OPTIONS])
    parse = serving.parse_questions
    if getattr(parse, "_mammoth", False):
        return

    @functools.wraps(parse)
    def parse_questions(request, limits):
        questions = parse(request, limits)
        for q in questions:
            spec = request.questions[q.id]
            object.__setattr__(q, "_mammoth_raw", (spec.instructions, spec.criteria))
        return questions

    parse_questions._mammoth = True  # type: ignore[attr-defined]
    serving.parse_questions = parse_questions


def patch_api() -> None:
    from .compat import api_router, strategies

    select = api_router.select_read_strategy
    if getattr(select, "_mammoth", False):
        return

    @functools.wraps(select)
    def select_read_strategy(model_config, supported_tasks):
        if not _is_mammoth(model_config):
            return select(model_config, supported_tasks)
        from .strategy import MammothStrategy
        try:
            if "generate" not in supported_tasks:
                raise ValueError("mammoth-1 structured decisions need the generate task")
            if model_config.logprobs_mode != "raw_logprobs":
                raise ValueError(f"mammoth-1 needs --logprobs-mode raw_logprobs, not {model_config.logprobs_mode}")
            _activate_question_rules()
        except ValueError as e:
            logger.error("/v1/systemone for mammoth-1 is disabled: %s", e)
            raise
        return _LoudStrategy.wrap(MammothStrategy)

    select_read_strategy._mammoth = True  # type: ignore[attr-defined]
    api_router.select_read_strategy = select_read_strategy
    strategies.select_read_strategy = select_read_strategy


class _LoudStrategy:
    """vLLM logs a strategy's init error at debug level only; make a misconfigured mammoth server say why."""

    @staticmethod
    def wrap(cls):
        init = cls.__init__
        if getattr(init, "_mammoth_loud", False):
            return cls

        @functools.wraps(init)
        def __init__(self, *a, **kw):
            try:
                init(self, *a, **kw)
            except ValueError as e:
                logger.error("/v1/systemone for mammoth-1 is disabled: %s", e)
                raise
            logger.info("/v1/systemone serves mammoth-1 with the block-bidirectional mask and the qev-labels-v2 "
                        "prompt (vllm-mammoth)")

        __init__._mammoth_loud = True  # type: ignore[attr-defined]
        cls.__init__ = __init__
        return cls
