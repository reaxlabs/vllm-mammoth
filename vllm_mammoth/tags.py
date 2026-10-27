# SPDX-License-Identifier: Apache-2.0
"""How a /v1/systemone read tells the model runner its block-0 length.

The API process and the engine/worker processes share nothing but the request, and the request id is the one
field every vLLM layer keeps (the engine only appends a suffix). The strategy appends ``-mmb<P>k<mac>`` to its
request id. ``mac`` is an HMAC over the id before the tag and P, keyed with a secret that ``register()`` puts into
the environment of the first vLLM process, so every engine and worker process vLLM spawns has the same key. A
client cannot forge a tag (for example through X-Request-Id), and an unsigned or wrongly signed tag is ignored: the
request keeps vLLM's causal mask.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets

KEY_ENV = "VLLM_MAMMOTH_TAG_KEY"
_TAG = re.compile(r"-mmb(\d+)k([0-9a-f]{16})(?=-|$)")
#: End of block 1: the rest of the sequence. A read samples one token and never feeds it back.
BLOCK1_END = (1 << 30) - 1


def ensure_key() -> None:
    """Create the shared tag key once, in the first process that loads the plugin (child processes inherit it).
    For multi-node or externally launched workers set VLLM_MAMMOTH_TAG_KEY to the same value everywhere."""
    if not os.environ.get(KEY_ENV):
        os.environ[KEY_ENV] = secrets.token_hex(32)


def _mac(prefix: str, prefix_len: int) -> str:
    key = os.environ.get(KEY_ENV, "").encode()
    if not key:
        raise RuntimeError(f"{KEY_ENV} is not set (vllm-mammoth was not loaded in this process)")
    return hmac.new(key, f"{prefix}|{prefix_len}".encode(), hashlib.sha256).hexdigest()[:16]


def tag_request_id(request_id: str, prefix_len: int) -> str:
    if prefix_len < 1:
        raise ValueError("block 0 needs at least one token")
    return f"{request_id}-mmb{prefix_len}k{_mac(request_id, prefix_len)}"


def prefix_len_of(req_id: str) -> int | None:
    """The block-0 length P of a request id with a valid tag (the last tag counts), else None."""
    last = None
    for m in _TAG.finditer(req_id):
        last = m
    if last is None or not os.environ.get(KEY_ENV):
        return None
    p = int(last.group(1))
    if p < 1 or not hmac.compare_digest(last.group(2), _mac(req_id[: last.start()], p)):
        return None
    return p


def ranges_for(prefix_len: int) -> list[tuple[int, int]]:
    """Inclusive [start, end] bidirectional ranges (vLLM mm_prefix convention): block 0 and block 1. Attention is
    (causal) OR (key in the query's range), which is exactly the two-block mask."""
    return [(0, prefix_len - 1), (prefix_len, BLOCK1_END)]


def merge_ranges(existing: dict[int, list[tuple[int, int]]] | None, req_ids) -> dict[int, list[tuple[int, int]]] | None:
    """Add the mammoth ranges for every tagged request (keyed by batch index) to vLLM's per-request range dict.
    Tagged requests are text-only, so their own entry is replaced, not extended (ranges must not overlap)."""
    out = None
    for idx, req_id in enumerate(req_ids):
        if req_id is None:
            continue
        p = prefix_len_of(req_id)
        if p is None:
            continue
        if out is None:
            out = dict(existing or {})
        out[idx] = ranges_for(p)
    return existing if out is None else out
