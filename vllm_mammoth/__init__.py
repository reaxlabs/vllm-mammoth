# SPDX-License-Identifier: Apache-2.0
"""vllm-mammoth: serve baselabs/mammoth-1 on vLLM's /v1/systemone exactly as the checkpoint was trained.

Loaded by vLLM in every process through the `vllm.general_plugins` entry point. For every model without
mammoth's `modeling_overrides` (block-bidirectional full attention) nothing changes.
"""
from __future__ import annotations

import logging

__version__ = "0.1.0"

logger = logging.getLogger("vllm.mammoth")
_DONE = False


def register() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True
    from . import patches, tags

    tags.ensure_key()
    patches.patch_model_config()
    gdn = patches.patch_gdn_batch_invariance()
    v2 = patches.patch_runner_v2()
    v1 = patches.patch_runner_v1()
    try:
        patches.patch_api()
        api = True
    except ImportError as e:  # an engine-only install without the OpenAI server is fine; anything else is not
        (logger.debug if "fastapi" in str(e) else logger.error)("vllm-mammoth: /v1/systemone not patched: %s", e)
        api = False
    logger.info("vllm-mammoth %s loaded (runner v2 hook: %s, v1 hook: %s, /v1/systemone: %s, GDN batch-invariant: %s)",
                __version__, v2, v1, api, gdn)
