# vllm-mammoth

A vLLM plugin that serves [baselabs/mammoth-1](https://huggingface.co/baselabs/mammoth-1) on vLLM's standard
`POST /v1/systemone` endpoint exactly as the model was trained. It works with a stock vLLM 0.32.x install and does
not fork vLLM.

Deutsche Kurzfassung: [unten](#deutsch).

## Why a plugin is needed

mammoth-1 declares in its `config.json` (`modeling_overrides`) that its full-attention layers use a
**block-bidirectional mask**: the prompt has two blocks, the system turn plus the `<state>` block, and the question.
Tokens see each other in both directions within a block, the question block also reads the state block, and the
state block never reads the question. The model also has its own prompt (qev-labels-v2) and readout (softmax over
the slot letters A, B, C, ... at the last position).

Stock vLLM starts the model, but reads it causally with its generic `/v1/systemone` prompt, so the probabilities
differ from the model as trained. vLLM ships this kind of model-specific support for Decision 3.0. This plugin adds
it for mammoth-1.

## Install

Either the pip package into your vLLM 0.32.x environment:

```bash
pip install --no-deps "vllm-mammoth @ git+https://github.com/reaxlabs/vllm-mammoth@v0.1.0"
```

or the ready image (stock `vllm/vllm-openai:v0.32.0` plus the plugin):

```bash
docker build -t vllm-mammoth:0.1.0 -f docker/Dockerfile .
```

vLLM loads the plugin automatically in every process (entry point `vllm.general_plugins`). Models without
mammoth's `modeling_overrides` are not affected.

## Start parameters

```bash
hf download baselabs/mammoth-1 --revision 03007ab3c4481bf01c71ef63adbbbc4b2093d71a --local-dir /models/mammoth-1

VLLM_BATCH_INVARIANT=1 vllm serve /models/mammoth-1 \
  --served-model-name mammoth-1 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --logprobs-mode raw_logprobs \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN
```

With the image: `docker run --gpus all --ipc host -p 8000:8000 -v /models/mammoth-1:/models/mammoth-1:ro
vllm-mammoth:0.1.0` (same flags; extra vLLM flags can be appended).

| Flag | Why |
| --- | --- |
| `VLLM_BATCH_INVARIANT=1` | Same answer for the same request, whatever else is in the batch. |
| `--no-enable-chunked-prefill`, `--max-num-batched-tokens >= --max-model-len` | A prompt block must be computed in one step, or it loses its bidirectional attention. The plugin refuses to serve `/v1/systemone` otherwise. |
| `--logprobs-mode raw_logprobs` | The readout needs the unprocessed log-probabilities. Checked at start. |
| `--no-enable-prefix-caching` | vLLM 0.32 needs chunked prefill for prefix caching on this hybrid model, so it must be off. |
| `--max-model-len 16384` | The trained length. Longer requests are refused, never truncated. |
| `--max-num-seqs 256` | vLLM's default (1024) is more than the model's state cache holds on one 96 GB GPU. |
| `--attention-backend TRITON_ATTN` | The verified attention kernel (it supports the per-request bidirectional ranges and batch invariance). |

vLLM prints "This model does not officially support disabling chunked prefill" at start: expected, the plugin needs
it off. The plugin refuses `/v1/systemone` (logged as an ERROR) when a required flag is missing, on a vLLM other than
0.32.x, with prefix caching, speculative decoding, context parallelism, LoRA or full CUDA graphs for prefill batches.
On the first request it checks once that the engine really applies the mask (the same prompt with and without the
mask must differ); if not, every read is refused instead of answered with the wrong mask.

Multi-node or externally launched workers: install the plugin in every environment and set the same
`VLLM_MAMMOTH_TAG_KEY` (any random hex string) on all nodes; on one machine the plugin creates it itself.

Hardware: one GPU with at least 80 GB (bf16 weights are 51 GB). Verified on NVIDIA RTX PRO 6000 Blackwell Server
Edition (96 GB, driver 580). Other GPUs (H100, B200) are expected to work with the same flags but are not verified yet.

## Request

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "mammoth-1",
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund":  {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                             "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]}
  }
}'
```

The response is vLLM's standard `/v1/systemone` response (`answers` with `probabilities` per question).

Limits: up to 64 questions per request, up to 128 options per choice question, 2 to 10 levels per score question.
Top-level `instructions`, images and other media are refused: the model reads a fixed prompt, and this version is
text only. The `confidence` field is vLLM's own formula; the probabilities are the model's.

## Verification (27 Oct 2026, 1x RTX PRO 6000, vLLM 0.32.0)

Reference: the model authors' own runtime (OneJev qev DecisionEngine + the checkpoint's `serve.patch`, bf16).
620 requests / 817 questions: 300 items from the REAX subnet's validator item generator, 300 public dev-set items,
20 edge cases (JSON states, 64 questions, 128 options, a 12k-token state, ...).

| Check | Result | Gate |
| --- | --- | --- |
| Mean of the per-request maximum log-probability difference | 0.048 | <= 0.20 |
| Worst single difference | 0.77 (1 request above 0.50) | <= 2.0 |
| Same top answer as the reference | 99.4 to 99.6 % of questions (varies slightly between server starts) | > 99 % |
| Batch invariance: 20 requests alone vs. inside 52 concurrent requests, twice | bit-identical (max difference 0.0) | identical |
| Throughput, single-question reads, 64 in flight | about 32 decisions/s | - |

The remaining small differences come from different kernels in bf16 (for example, the reference computes the
answer head in fp32).

## How it works

- vLLM already supports per-request bidirectional token ranges (used for image tokens in Gemma models): attention =
  causal OR (key inside the query's range). The two ranges [0, P-1] and [P, end] give exactly mammoth's mask. The
  plugin switches this on for checkpoints whose `modeling_overrides` declare `block_bidirectional`, and its model
  runner hook gives each `/v1/systemone` read its ranges. The read's request id carries P with an HMAC signature
  (key shared by vLLM's processes), so a client cannot set the mask on its own requests through `X-Request-Id`.
- The block boundary is computed like the reference engine, including its consistency check: a request whose
  boundary is ambiguous (for example a state that quotes the internal prefix marker) is refused with 400, as the
  reference does.
- The `/v1/systemone` read strategy renders the authors' qev-labels-v2 prompt, computes the block boundary the same
  way as the reference engine, asks for the slot-letter log-probabilities and applies the slot_lm softmax.
- vLLM 0.32 refuses `VLLM_BATCH_INVARIANT=1` for every GatedDeltaNet model (Qwen3.5 family). A mammoth read is
  prefill only; the plugin declares batch invariance for the GDN layers and uses the Triton GDN prefill kernels,
  which were measured bit-identical across batches. (The FlashInfer GDN kernel vLLM picks by default was not.)
  This applies only when the served model is mammoth-1, and was verified for `/v1/systemone` reads (prefill only).
  Opt out: `VLLM_MAMMOTH_GDN_BATCH_INVARIANT=0`.

Licence: Apache-2.0. The prompt/readout port follows OneJev qev @81ce62f (Apache-2.0, the OneJev authors) and the
checkpoint's `serve.patch`. mammoth-1 itself is MIT (Base Labs, Baseten).

---

## Deutsch

**Was es ist:** Ein vLLM-Plugin, mit dem ein normales vLLM 0.32.x das Modell baselabs/mammoth-1 auf dem
Standard-Endpunkt `POST /v1/systemone` genau so beantwortet, wie das Modell trainiert wurde. Kein vLLM-Fork.

**Warum:** mammoth-1 ist mit einer eigenen Attention-Maske trainiert (Eingabe- und Frageteil werden jeweils in beide
Richtungen gelesen) und hat einen eigenen Prompt und eine eigene Auslese. Standard-vLLM startet das Modell, rechnet
aber kausal und mit seinem allgemeinen Prompt, die Wahrscheinlichkeiten stimmen dann nicht. Für Decision 3.0 bringt
vLLM so eine modellspezifische Unterstützung schon mit; das Plugin ergänzt sie für mammoth-1.

**Installation:** `pip install --no-deps "vllm-mammoth @ git+https://github.com/reaxlabs/vllm-mammoth@v0.1.0"` in die
vLLM-Umgebung, oder das fertige Image (`docker/Dockerfile`: Standard-Image `vllm/vllm-openai:v0.32.0` plus Plugin).
vLLM lädt das Plugin automatisch; andere Modelle bleiben unverändert.

**Startparameter:** siehe oben ("Start parameters"). Wichtig: `VLLM_BATCH_INVARIANT=1`, `--no-enable-chunked-prefill`,
`--max-num-batched-tokens 16384`, `--max-model-len 16384`, `--logprobs-mode raw_logprobs`,
`--no-enable-prefix-caching`, `--max-num-seqs 256`, `--attention-backend TRITON_ATTN`. Eine GPU mit mindestens 80 GB,
geprüft auf RTX PRO 6000 Blackwell Server Edition (96 GB). Fehlt ein nötiger Parameter, verweigert das Plugin
/v1/systemone mit einer klaren Fehlermeldung im Log, statt falsch zu rechnen.

**Prüfung:** Gegen die Original-Laufzeit der Modellautoren auf 620 Anfragen (817 Fragen): mittlere Abweichung 0,048
(Grenze 0,20), größte 0,77 (Grenze 2,0), gleiche Top-Antwort bei 99,4 bis 99,6 % der Fragen; mit `VLLM_BATCH_INVARIANT=1`
bitgleiche Ergebnisse, egal ob eine Anfrage allein oder zwischen 52 parallelen Anfragen läuft.

**Grenzen:** bis 64 Fragen pro Anfrage, bis 128 Optionen, 2 bis 10 Stufen; keine Top-Level-`instructions`, keine
Bilder (diese Version ist reiner Text); zu lange Anfragen werden abgelehnt, nie gekürzt.
