# SPDX-License-Identifier: Apache-2.0
"""/v1/systemone read strategy for baselabs/mammoth-1 (qev-labels-v2 prompt, slot_lm readout)."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import math
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

from . import qev
from .compat import (
    DecisionLimits,
    QuestionRead,
    ReadContext,
    ReadStrategy,
    SamplingParams,
    StructuredDecisionError,
    label_softmax,
    next_token_label_reads,
    tokens_input,
)
from .tags import tag_request_id

logger = logging.getLogger("vllm.mammoth")
TESTED_VLLM = ("0.32.",)

#: vLLM's per-request cap on logprob_token_ids (sampling_params.MAX_LOGPROB_TOKEN_IDS in 0.32).
MAX_OPTIONS = 128
MAX_QUESTIONS = 64

# A per-process secret: prefix-cache entries computed under the mammoth mask are salted with it, so no request
# from outside this strategy can share (or poison) them.
_SALT_KEY = secrets.token_bytes(32)


def _salt(client_salt: str | None, ids: Sequence[int], prefix_len: int) -> str:
    msg = f"{client_salt or ''}|{prefix_len}|{','.join(map(str, ids))}".encode()
    return "mammoth-" + hmac.new(_SALT_KEY, msg, hashlib.sha256).hexdigest()


class MammothStrategy(ReadStrategy):
    """baselabs/mammoth-1. Each question is one generate request of the prompt the checkpoint was trained with
    (system turn + <state> block, then the question with its slot-lettered options). The request id carries the
    block-0 length P, and the plugin's runner hook turns it into the two bidirectional ranges of the checkpoint's
    block-bidirectional mask. The probabilities are the softmax over the slot tokens' logits at the last position
    (slot_lm, temperature 1)."""

    task = "generate"
    modalities = frozenset()

    def __init__(self, context: ReadContext):
        super().__init__(context)
        model_config = context.engine_client.model_config
        self.tokenizer = context.online_renderer.renderer.get_tokenizer()
        self.slot_ids = qev.slot_ids(self.tokenizer)[:MAX_OPTIONS]
        self.max_model_len = model_config.max_model_len
        self.temperature = 1.0  # no calibration.json in the pinned revision
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("mammoth-1 needs the fast tokenizer (character offsets for the block boundary check)")
        self._check_server(context)
        self._mask_ok: bool | None = None
        self._mask_lock: asyncio.Lock | None = None

    @staticmethod
    def _check_server(context: ReadContext) -> None:
        import vllm

        cfg = context.engine_client.vllm_config
        sched = cfg.scheduler_config
        version = getattr(vllm, "__version__", "?")
        if not version.startswith(TESTED_VLLM):
            raise ValueError(f"vllm-mammoth 0.1 is verified on vLLM 0.32.x, this is {version}")
        if getattr(sched, "enable_chunked_prefill", False):
            raise ValueError(
                "mammoth-1 needs --no-enable-chunked-prefill: a prompt block split across scheduler steps would "
                "lose its bidirectional attention")
        if sched.max_num_batched_tokens < cfg.model_config.max_model_len:
            raise ValueError(
                f"mammoth-1 needs --max-num-batched-tokens >= --max-model-len "
                f"({sched.max_num_batched_tokens} < {cfg.model_config.max_model_len})")
        if not cfg.model_config.is_mm_prefix_lm:
            raise ValueError("mammoth-1 block mask is not active (is_mm_prefix_lm is False); is the plugin loaded "
                             "in the engine process?")
        if getattr(cfg.cache_config, "enable_prefix_caching", False):
            raise ValueError("mammoth-1 needs --no-enable-prefix-caching")
        mode = getattr(cfg.compilation_config, "cudagraph_mode", None)
        if mode is not None:
            from vllm.config.compilation import CUDAGraphMode

            if mode.mixed_mode() == CUDAGraphMode.FULL:
                raise ValueError("mammoth-1 cannot use full CUDA graphs for prefill batches (the block ranges are "
                                 "not captured); use the default cudagraph mode or FULL_AND_PIECEWISE")
        if cfg.speculative_config is not None:
            raise ValueError("mammoth-1 reads do not support speculative decoding (untested with the block mask)")
        par = cfg.parallel_config
        if getattr(par, "decode_context_parallel_size", 1) > 1 or getattr(par, "prefill_context_parallel_size", 1) > 1:
            raise ValueError("mammoth-1 reads do not support context parallelism (untested with the block mask)")
        if cfg.lora_config is not None:
            raise ValueError("mammoth-1 reads do not support LoRA adapters (untested with the block mask)")

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=MAX_QUESTIONS, max_options=MAX_OPTIONS)

    def render_state(self, state: Any) -> str:
        return qev.render_state(state)

    def _branch(self, q) -> qev.Branch:
        raw = getattr(q, "_mammoth_raw", None)
        kind = q.type.name
        if raw is not None:  # the request's own JSON values (see patches.parse_questions)
            instructions, criteria = raw
        else:
            instructions, criteria = q.instructions, None
        try:
            if kind == "noul":
                crit = criteria if isinstance(criteria, Mapping) else {}
                if raw is None:
                    yes, no = q.options
                    descs = [yes.description, no.description]
                else:
                    descs = [crit.get("true"), crit.get("false")]
                return qev.render_question("noul", instructions, ["yes", "no"], descs)
            if kind == "choice":
                names = [o.name for o in q.options]
                if raw is not None and isinstance(criteria, Mapping):
                    descs = [criteria[n] for n in names]
                else:
                    descs = [o.description for o in q.options]
                return qev.render_question("choice", instructions, names, descs)
            if kind == "score":
                levels = list(criteria) if raw is not None and isinstance(criteria, list) else [o.name for o in q.options]
                return qev.render_question("score", instructions, [], levels)
        except qev.ContractError as e:
            raise StructuredDecisionError(f"question {q.id!r}: {e}") from e
        raise StructuredDecisionError(f"question {q.id!r}: this model does not answer {kind!r} questions")

    async def read(
        self,
        questions: list,
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        if instructions:
            raise StructuredDecisionError("this model reads a fixed prompt, so it takes no instructions")
        if (chat_template_kwargs or {}).get("enable_thinking"):
            raise StructuredDecisionError("a read is the reply's first token, so thinking must be off")
        branches = [self._branch(q) for q in questions]
        for q, b in zip(questions, branches):
            if len(b.keys) > MAX_OPTIONS:
                raise StructuredDecisionError(f"question {q.id!r}: at most {MAX_OPTIONS} options for this model")
        if len(self.slot_ids) < max(len(b.keys) for b in branches):
            raise StructuredDecisionError(f"this tokenizer has only {len(self.slot_ids)} answer slots")
        try:
            full, prefix_len = await asyncio.to_thread(
                qev.encode_request, self.tokenizer, state, [b.suffix for b in branches], check_layout=True)
        except qev.ContractError as e:
            raise StructuredDecisionError(str(e)) from e
        await self._ensure_mask_active(lora_request, priority)
        for q, ids in zip(questions, full):
            if len(ids) > self.max_model_len:
                raise StructuredDecisionError(
                    f"question {q.id!r}: the prompt has {len(ids)} tokens, more than max_model_len="
                    f"{self.max_model_len}; nothing is truncated")

        engine_inputs = [tokens_input(ids, cache_salt=_salt(cache_salt, ids, prefix_len)) for ids in full]
        params = [
            SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=self.slot_ids[: len(b.keys)])
            for b in branches
        ]
        # One tag for the request: every question shares block 0 (P tokens); block 1 is the rest of each prompt.
        label_reads = await next_token_label_reads(
            self.context.engine_client,
            engine_inputs,
            params,
            tag_request_id(request_id, prefix_len),
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )

        return self._reads(branches, full, label_reads)

    def _reads(self, branches, full, label_reads) -> list[QuestionRead]:
        reads = []
        for b, ids, label_read in zip(branches, full, label_reads):
            output = label_read.result.outputs[0]
            lps = label_read.logprobs
            probs = label_softmax([lp / self.temperature for lp in lps])
            slot_set = set(self.slot_ids[: len(b.keys)])
            reads.append(QuestionRead(
                probs=probs,
                label_mass=sum(math.exp(lp) for lp in lps),
                argmax_is_label=bool(output.token_ids) and output.token_ids[0] in slot_set,
                input_tokens=len(ids),
                output_tokens=len(output.token_ids),
            ))
        return reads

    async def _ensure_mask_active(self, lora_request, priority: int) -> None:
        """Once per server: the same prompt read with and without the block-mask tag must differ. If it does not,
        the engine/worker processes do not apply the mask (plugin missing there, unsupported vLLM build), and the
        server refuses every read instead of answering causally."""
        if self._mask_ok:
            return
        if self._mask_lock is None:
            self._mask_lock = asyncio.Lock()
        async with self._mask_lock:
            if self._mask_ok is None:
                self._mask_ok = await self._probe_mask(lora_request, priority)
                if self._mask_ok:
                    logger.info("vllm-mammoth: block mask verified in the engine")
                else:
                    logger.error("vllm-mammoth: the engine ignores the block mask; /v1/systemone refuses reads. Is "
                                 "vllm-mammoth installed in every engine/worker environment (and %s identical on "
                                 "all nodes)?", "VLLM_MAMMOTH_TAG_KEY")
        if not self._mask_ok:
            raise RuntimeError("mammoth-1 block mask is not active in the engine; refusing to answer causally")

    async def _probe_mask(self, lora_request, priority: int) -> bool:
        branch = qev.render_question("choice", "Which team should handle this?", ["billing", "technical", "fraud"],
                                     ["Charges, refunds, invoices", "App or site faults", "Suspected unauthorised use"])
        full, p = qev.encode_request(self.tokenizer, "I was charged twice this month, please refund one of them.",
                                     [branch.suffix])
        ids = full[0]
        params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=self.slot_ids[:3])
        rid = "mammoth-maskcheck-" + secrets.token_hex(6)
        masked, causal = await asyncio.gather(
            next_token_label_reads(self.context.engine_client, [tokens_input(ids, cache_salt=_salt(None, ids, p))],
                                   [params], tag_request_id(rid + "-a", p), lora_request=lora_request,
                                   priority=priority),
            next_token_label_reads(self.context.engine_client, [tokens_input(ids, cache_salt=_salt("causal", ids, 0))],
                                   [params], rid + "-b", lora_request=lora_request, priority=priority))
        return masked[0].logprobs != causal[0].logprobs
