#!/usr/bin/env bash
# Start parameters for baselabs/mammoth-1 with vllm-mammoth (see README). Extra vLLM flags can be appended.
set -euo pipefail
exec vllm serve "${MAMMOTH_MODEL:-/models/mammoth-1}" \
  --served-model-name "${MAMMOTH_SERVED_NAME:-mammoth-1}" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --logprobs-mode raw_logprobs \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  "$@"
