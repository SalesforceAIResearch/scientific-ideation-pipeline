"""Shared prompt package.

Single source of truth for every prompt in the project, imported by both
inference (`inference/`) and the Part-B dataset generator
(`dataset_creation/part_b/summarize_document.py`):

  ideation.py       the v0 inference prompt + document-block builders
  summarization.py  the Part-B summary prompts, the SummaryType enum, and the
                    SUMMARY_SPECS registry

Re-exported here so callers can `from prompts import IDEATION_V0_TEMPLATE,
SummaryType, build_prompt, ...` regardless of which submodule a symbol lives in.

NOTE on the older standalone `inference/prompts.py` and `strategy_check/
prompts.py`: scripts that put their OWN directory on sys.path[0] still resolve
their local `prompts` first; this package is found by scripts that put the repo
ROOT on the path.
"""

from .ideation import (
    IDEATION_V0_TEMPLATE,
    IDEATION_SI_TEMPLATE,
    DEFAULT_NUM_IDEAS,
    ANSWER_FIELDS,
    SI_ANSWER_FIELDS,
    PROMPT_TEMPLATES,
    PROMPT_TYPES,
    DEFAULT_PROMPT_TYPE,
    PROMPT_ANSWER_FORMAT,
    format_documents,
    build_prompt,
)
from .summarization import (
    Scope,
    SummaryType,
    SUMMARY_SPECS,
    ALL_SUMMARY_TYPES,
    summary_type,
    # prompt text (exported for tests / external inspection)
    IDEA_SYSTEM,
    IDEA_TEMPLATE,
    TARGET_IDEA_TEMPLATE,
    FULL_SUMM_SYSTEM,
    FULL_SUMM_TEMPLATE,
    IDEA_GROUNDED_SYSTEM,
    IDEA_GROUNDED_TEMPLATE,
    SOURCE_MULTIPLE_OUTPUT_TEMPLATE,
    SI_SOURCE_SUMM_SYSTEM,
    SI_SOURCE_SUMM_TEMPLATE,
    SI_IDEA_GROUNDED_SYSTEM,
    SI_IDEA_GROUNDED_TEMPLATE,
)

__all__ = [
    "IDEATION_V0_TEMPLATE",
    "IDEATION_SI_TEMPLATE",
    "DEFAULT_NUM_IDEAS",
    "ANSWER_FIELDS",
    "SI_ANSWER_FIELDS",
    "PROMPT_TEMPLATES",
    "PROMPT_TYPES",
    "DEFAULT_PROMPT_TYPE",
    "PROMPT_ANSWER_FORMAT",
    "format_documents",
    "build_prompt",
    "Scope",
    "SummaryType",
    "SUMMARY_SPECS",
    "ALL_SUMMARY_TYPES",
    "summary_type",
    "IDEA_SYSTEM",
    "IDEA_TEMPLATE",
    "TARGET_IDEA_TEMPLATE",
    "FULL_SUMM_SYSTEM",
    "FULL_SUMM_TEMPLATE",
    "IDEA_GROUNDED_SYSTEM",
    "IDEA_GROUNDED_TEMPLATE",
    "SOURCE_MULTIPLE_OUTPUT_TEMPLATE",
    "SI_SOURCE_SUMM_SYSTEM",
    "SI_SOURCE_SUMM_TEMPLATE",
    "SI_IDEA_GROUNDED_SYSTEM",
    "SI_IDEA_GROUNDED_TEMPLATE",
]
