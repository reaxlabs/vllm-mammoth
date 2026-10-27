"""CPU tests on the real mammoth-1 tokenizer. MAMMOTH_TOKENIZER_DIR = a directory with config.json, tokenizer.json,
tokenizer_config.json and chat_template.jinja of baselabs/mammoth-1@03007ab3. Optional QEV_REFERENCE_PORT = path to an
independent port of the qev prompt contract (normalize_question / encode_request / slot_ids) to compare against."""
import importlib.util
import json
import os

import pytest

from vllm_mammoth import qev, tags

TOK_DIR = os.environ.get("MAMMOTH_TOKENIZER_DIR", "")
LANE_T = os.environ.get("QEV_REFERENCE_PORT", "")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    if not os.path.exists(os.path.join(TOK_DIR, "tokenizer.json")):
        pytest.skip("no mammoth tokenizer files")
    return transformers.AutoTokenizer.from_pretrained(TOK_DIR)


@pytest.fixture(scope="module")
def ref():
    if not LANE_T or not os.path.exists(LANE_T):
        pytest.skip("QEV_REFERENCE_PORT not set")
    spec = importlib.util.spec_from_file_location("qev_contract_ref", LANE_T)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["qev_contract_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


CASES = [
    ("I was charged twice this month, please refund one of them.", {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                              "fraud": "Suspected unauthorised use"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Can wait", "Today", "Blocking the customer now"]},
    }),
    ({"ticket": 42, "text": "Ünïcödé — test", "tags": ["a", "b"]}, {
        "q": {"type": "choice", "instructions": {"ask": "pick one", "lang": "de"},
              "criteria": {"x": None, "y": {"why": "nested"}, "z": "  padded  "}},
        "n": {"type": "noul", "criteria": {"true": "yes it is", "false": ""}},
        "s": {"type": "score", "criteria": [{"lvl": 0}, "mid", "high"]},
    }),
    ("", {"only": {"type": "noul"}}),
    ("many options", {"big": {"type": "choice", "instructions": "Which code?",
                              "criteria": {f"opt{i}": f"description {i}" for i in range(128)}}}),
]


def _ours(tok, state, questions):
    branches = []
    for spec in questions.values():
        kind, crit = spec["type"], spec.get("criteria")
        if kind == "noul":
            crit = crit or {}
            b = qev.render_question("noul", spec.get("instructions"), ["yes", "no"], [crit.get("true"), crit.get("false")])
        elif kind == "choice":
            b = qev.render_question("choice", spec.get("instructions"), list(crit), [crit[k] for k in crit])
        else:
            b = qev.render_question("score", spec.get("instructions"), [], list(crit))
        branches.append(b)
    full, p = qev.encode_request(tok, qev.render_state(state), [b.suffix for b in branches], check_layout=True)
    return branches, full, p


@pytest.mark.parametrize("state,questions", CASES)
def test_render_matches_reference_port(tok, ref, state, questions):
    branches, full, p = _ours(tok, state, questions)
    rq = [ref.normalize_question(q) for q in questions.values()]
    assert [b.suffix for b in branches] == [q.suffix for q in rq]
    assert [list(b.keys) for b in branches] == [q.keys for q in rq]
    rfull, rp = ref.encode_request(tok, state, rq)
    assert full == rfull and p == rp
    # block 0 ends right after the state block
    text = tok.decode(full[0][:p])
    assert text.rstrip().endswith("</state>")


def test_slot_ids_match_reference(tok, ref):
    assert qev.slot_ids(tok) == ref.slot_ids(tok)
    assert len(qev.slot_ids(tok)) == 256
    assert qev.SLOT_LABELS == ref.SLOT_LABELS


def test_tags(monkeypatch):
    monkeypatch.setenv(tags.KEY_ENV, "ab" * 32)
    rid = tags.tag_request_id("decision-abc", 57) + "-3" + "-deadbeef"
    assert tags.prefix_len_of(rid) == 57
    assert tags.prefix_len_of("chatcmpl-123") is None
    # a client-supplied tag without the key's MAC is ignored
    assert tags.prefix_len_of("chatcmpl-evil-mmb5k0123456789abcdef-1a2b3c4d") is None
    forged = tags.tag_request_id("decision-abc", 57).replace("mmb57", "mmb58")
    assert tags.prefix_len_of(forged) is None
    # a user tag inside the base id does not override ours (last tag counts, MAC binds the prefix)
    good = tags.tag_request_id("decision-user-mmb9k0000000000000000", 12) + "-0-ab12cd34"
    assert tags.prefix_len_of(good) == 12
    merged = tags.merge_ranges({0: [(3, 7)]}, ["chatcmpl-1", tags.tag_request_id("decision-a", 10) + "-0-ffff0000", None])
    assert merged == {0: [(3, 7)], 1: [(0, 9), (10, tags.BLOCK1_END)]}
    assert tags.merge_ranges(None, ["a", "b"]) is None
    monkeypatch.setenv(tags.KEY_ENV, "cd" * 32)  # another server's key
    assert tags.prefix_len_of(rid) is None


def test_marker_in_state_is_refused(tok):
    b = qev.render_question("noul", "Is it?", ["yes", "no"], [None, None])
    with pytest.raises(qev.ContractError):
        qev.encode_request(tok, "a state that quotes " + qev.PREFIX_MARK + " inside", [b.suffix], check_layout=True)


def test_overrides_detection():
    cfg = json.load(open(os.path.join(TOK_DIR, "config.json"))) if os.path.exists(os.path.join(TOK_DIR, "config.json")) else None
    if cfg is None:
        pytest.skip("no config.json")
    assert qev.mammoth_overrides(cfg) is not None
    assert qev.mammoth_overrides({"architectures": ["Qwen3_5ForConditionalGeneration"]}) is None
