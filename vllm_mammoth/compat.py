# SPDX-License-Identifier: Apache-2.0
"""vLLM imports the plugin uses, in one place (module paths of vLLM 0.32.x; newer layouts as fallback)."""
from __future__ import annotations

import importlib

_PKGS = (
    "vllm.entrypoints.generate.structured_decisions",  # vLLM 0.32.x
    "vllm.entrypoints.openai.decisions",  # vLLM main after 0.32 (module moved)
)


def _find(sub: str):
    errors = []
    for pkg in _PKGS:
        try:
            return importlib.import_module(f"{pkg}.{sub}")
        except ImportError as e:
            errors.append(f"{pkg}.{sub}: {e}")
    raise ImportError("vLLM structured decisions (/v1/systemone) not found: " + "; ".join(errors))


strategies = _find("strategies")
question_types = _find("question_types")
serving = _find("serving")
api_router = _find("api_router")

ReadStrategy = strategies.ReadStrategy
ReadContext = strategies.ReadContext
DecisionLimits = strategies.DecisionLimits
QuestionRead = strategies.QuestionRead
StructuredDecisionError = question_types.StructuredDecisionError
label_softmax = question_types.label_softmax

from vllm.entrypoints.generate.label_reads import next_token_label_reads  # noqa: E402
from vllm.inputs import tokens_input  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
