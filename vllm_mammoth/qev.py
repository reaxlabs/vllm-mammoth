# SPDX-License-Identifier: Apache-2.0
"""The prompt and readout contract of baselabs/mammoth-1 (qev-labels-v2), without vLLM imports.

Ported from OneJev qev @81ce62f + the checkpoint's serve.patch (qev/prompt.py, qev/answers.py, qev/engine.py
`_prefix_and_suffix_ids`; Apache-2.0, (c) the OneJev authors). Same rendered text, token ids, prefix/suffix split
and slot labels as the authors' DecisionEngine.

Attention contract (config.json `modeling_overrides`): full-attention layers use a two-block mask. Block 0 is the
system turn plus the `<state>` block (the shared prefix, P tokens), block 1 is the question suffix. Tokens attend
bidirectionally within their block, block 1 also reads block 0, block 0 never reads block 1. GatedDeltaNet layers
stay causal.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROMPT_VERSION = "qev-labels-v2"
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                  "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
SLOT_LABELS = tuple(list(LETTERS) + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - 26])
MAX_SCORE_LEVELS = 10
NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"
CHOICE_DEFAULT_INSTRUCTIONS = "Which option applies to the state?"
SCORE_DEFAULT_INSTRUCTIONS = "Which level describes the state?"
QUESTION_HEADER, SCORE_HEADER = "Question:", "Rate the state:"
ANSWER_LINE, ANSWER_LINE_MANY = "Answer with one letter: {letters}.", "Answer with one label: {letters}."
PREFIX_MARK = "QEV_PREFIX_MARK"


class ContractError(ValueError):
    """A request the model's contract cannot express (returned as HTTP 400)."""


def render_entry(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def state_block(state_text: str) -> str:
    return f"<state>\n{state_text}\n</state>\n\n"


def slot_ids(tokenizer) -> list[int]:
    """Token ids of the slot labels (qev DecisionEngine._verify_slots): A..Z required, two-letter labels
    all-or-nothing."""
    def one(label):
        enc = tokenizer.encode(label, add_special_tokens=False)
        return enc[0] if len(enc) == 1 and tokenizer.decode(enc) == label else None

    ids = [one(label) for label in LETTERS]
    if any(i is None for i in ids):
        raise ValueError("answer slots A..Z are not single round-trip tokens")
    extra = [one(label) for label in SLOT_LABELS[26:]]
    if all(i is not None for i in extra):
        ids += extra
    if len(set(ids)) != len(ids):
        raise ValueError("answer slot tokens collide")
    return ids


@dataclass(frozen=True)
class Branch:
    kind: str  # noul | choice | score
    keys: tuple[str, ...]  # answer keys in slot order (noul: yes, no)
    suffix: str


def render_question(kind: str, instructions: Any, names: Sequence[str], descriptions: Sequence[Any]) -> Branch:
    """One question's suffix (qev render_question, tagged layout). ``names``/``descriptions`` in slot order:
    noul -> yes/no with the true/false descriptions, choice -> option keys, score -> levels lowest first."""
    if kind == "noul":
        text = render_entry(instructions) or NOUL_DEFAULT_INSTRUCTIONS
        t, f = descriptions
        descs = [NOUL_DEFAULT_TRUE if t is None else render_entry(t),
                 NOUL_DEFAULT_FALSE if f is None else render_entry(f)]
        keys, shown = ("yes", "no"), ["yes", "no"]
    elif kind == "choice":
        text = render_entry(instructions) or CHOICE_DEFAULT_INSTRUCTIONS
        keys, shown, descs = tuple(names), list(names), [render_entry(d) for d in descriptions]
    elif kind == "score":
        if not 1 <= len(descriptions) <= MAX_SCORE_LEVELS:
            raise ContractError(f"a score has 1 to {MAX_SCORE_LEVELS} levels")
        text = render_entry(instructions) or SCORE_DEFAULT_INSTRUCTIONS
        keys = tuple(str(i) for i in range(len(descriptions)))
        shown, descs = [f"level {i}" for i in range(len(descriptions))], [render_entry(d) for d in descriptions]
    else:
        raise ContractError(f"unsupported question type {kind!r} (choice, noul or score)")
    if len(keys) > len(SLOT_LABELS):
        raise ContractError(f"at most {len(SLOT_LABELS)} options")
    letters = list(SLOT_LABELS[: len(keys)])
    lines = [f"{letter}. {name}: {desc}" if desc else f"{letter}. {name}"
             for letter, name, desc in zip(letters, shown, descs)]
    head = f"{QUESTION_HEADER if kind != 'score' else SCORE_HEADER} {text}\n\nOptions:\n"
    answer = ANSWER_LINE if len(letters) <= 26 else ANSWER_LINE_MANY
    return Branch(kind, keys, head + "\n".join(lines) + "\n\n" + answer.format(letters=", ".join(letters)))


def render(tokenizer, state_text: str, suffix: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": state_block(state_text) + suffix}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_request(tokenizer, state_text: str, suffixes: Sequence[str],
                   check_layout: bool = False) -> tuple[list[list[int]], int]:
    """Token ids of every branch and the shared-prefix length P (qev _prefix_and_suffix_ids): the longest common
    token prefix of all branches that ends at or before the end of the state block. P is the block-0 length.

    ``check_layout`` adds the reference engine's block-boundary check (serve.patch DecisionEngine._split): per
    branch, the tokens that end at or before the suffix's last occurrence in the prompt must be exactly P, else the
    request is refused (e.g. a state that quotes the prefix marker)."""
    prompts = [render(tokenizer, state_text, s) for s in suffixes]
    full = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    text = render(tokenizer, state_text, PREFIX_MARK)
    prefix = tokenizer.encode(text[: text.index(PREFIX_MARK)], add_special_tokens=False)
    n = len(prefix)
    while n > 0 and not all(ids[:n] == prefix[:n] for ids in full):
        n -= 1
    if n <= 0 or any(len(ids) <= n for ids in full):
        raise ContractError("a question has an empty suffix (state block boundary mismatch)")
    if check_layout:
        import bisect

        for p, s, ids in zip(prompts, suffixes, full):
            enc = tokenizer(p, add_special_tokens=False, return_offsets_mapping=True)
            if list(enc["input_ids"]) != ids:
                raise ContractError("layout tokenization differs from the branch tokenization")
            ends = [e for _, e in enc["offset_mapping"]]
            if bisect.bisect_right(ends, p.rindex(s)) != n:
                raise ContractError("the state block boundary is ambiguous for this request")
    return full, n


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    import math
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    w = [math.exp(x - top) for x in scaled]
    s = sum(w)
    return [x / s for x in w]


# System One confidence formulas (qev answers.py), for consumers that need them from the probabilities.
def choice_confidence(p: Sequence[float]) -> float:
    if len(p) == 1:
        return 1.0
    u = 1.0 / len(p)
    return (max(p) - u) / (1.0 - u)


def score_confidence(p: Sequence[float]) -> float:
    if len(p) == 1:
        return 1.0
    k = len(p)
    mode = max(range(k), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (k - 1) / 2
    return max(0.0, 1.0 - dist / (sum(abs(i - c) for i in range(k)) / k))


def mammoth_overrides(hf_config: Any) -> Mapping[str, Any] | None:
    """The checkpoint's modeling_overrides when they declare the block-bidirectional mask, else None."""
    ov = getattr(hf_config, "modeling_overrides", None)
    if ov is None and isinstance(hf_config, Mapping):
        ov = hf_config.get("modeling_overrides")
    if not isinstance(ov, Mapping):
        return None
    att = ov.get("attention") or {}
    if att.get("full_attention") != "block_bidirectional":
        return None
    if att.get("linear_attention", "causal") != "causal":
        raise ValueError(f"unsupported linear_attention mask {att.get('linear_attention')!r}")
    if ov.get("readout", "slot_lm") != "slot_lm":
        raise ValueError(f"unsupported readout {ov.get('readout')!r} (the plugin reads slot_lm)")
    return ov
