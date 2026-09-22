"""Parse the policy's `<answer>` block into a list of structured ideas.

This is the INPUT ADAPTER shared by the evaluators (eval/, evals/) and the
F9-B reward (rewards/f9b.py, later). The ideation policy emits a JSON *list* of
K idea objects inside `<answer>...</answer>`, per the schema in
`prompts/ideation.py` (`IDEATION_V0_TEMPLATE`):

    [
      {
        "motivation": "...",
        "problem": "...",
        "insight": {
          "claim": "...",
          "grounding": [{"source": "...", "supports": "..."}],
          "grounding_type": "in_domain_extension" | ... | "reframing"
        }
      },
      ... (K of them)
    ]

`inference/parsing.py::parse_answer_fields()` only pulls the FIRST `{...}` (a
single idea) and flattens it for display -- it does NOT handle the list. This
module is the list-aware parser. We reuse `inference/parsing.py::extract_answer_str`
for the robust `<answer>` / unclosed-tag / `</think>`-tail extraction, then
`json.loads` the list and validate each element.

The `idea_to_*` adapters map an `Idea` onto the card shapes the rubric judges
expect (novelty/grounding render `{problem, insight, motivation}` cards or
`{title, abstract}` references), so the evaluators accept either a pre-built
eval-set dataset OR raw policy output via this module.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo root importable so `from prompts import ...` and
# `inference.parsing` resolve whether this is run as a script, imported by an
# eval entry point, or loaded by verl via custom_reward_function.path.
_GRPO_ROOT = Path(__file__).resolve().parents[1]      # .../grpo_training
_REPO_ROOT = _GRPO_ROOT.parent                        # repo root (has prompts/, inference/)
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the robust answer-region extractor from inference (handles closed
# <answer>...</answer>, unclosed <answer>..., and the </think>-tail shape that
# reasoning models like Olmo-3-Think emit).
try:
    from inference.parsing import extract_answer_str  # type: ignore
    # Balanced-span scanner for the bare concatenated object stream
    # (`{...}{...}{...}`, no [] / commas) that _loads_list falls back to; see the
    # comment there. Falls back to a no-op if inference isn't importable, so
    # _loads_list simply skips the stream branch and behaves as before.
    from inference.parsing import top_level_idea_objects as _top_level_idea_objects  # type: ignore
except Exception:  # pragma: no cover - fallback if inference isn't importable
    def _top_level_idea_objects(answer_str, idea_keys):  # type: ignore
        return []
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    _ANSWER_OPEN_RE = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)
    _THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

    def extract_answer_str(raw: str) -> Optional[str]:  # type: ignore
        if not raw:
            return None
        m = _ANSWER_RE.search(raw)
        if m:
            return m.group(1).strip()
        m = _ANSWER_OPEN_RE.search(raw)
        if m:
            return m.group(1).strip()
        parts = _THINK_CLOSE_RE.split(raw)
        if len(parts) > 1:
            return parts[-1].strip()
        return raw.strip()


# Optional: json_repair salvages idea lists from malformed JSON. Lightly-trained
# checkpoints routinely emit ONE broken object mid-array (literal control chars in
# a string, a trailing comma, a `// comment`, an unterminated string) which strict
# json.loads rejects WHOLESALE -- dropping all K ideas and hard-failing the format
# gate. This is the SAME fallback inference/parsing.py already uses; ported here so
# the reward's parser is no longer stricter than the eval parser. Absent -> we
# silently skip the repair fallback (identical behaviour to before).
try:
    from json_repair import repair_json as _repair_json  # type: ignore
except Exception:  # pragma: no cover
    _repair_json = None

# Idea-shaped keys used to filter json_repair's occasional coerced trailing
# fragment (mirrors inference/parsing.py::_IDEA_KEYS).
_IDEA_KEYS = {"motivation", "problem", "insight"}

# Default number of ideas per generation (K in the reward doc). Imported from
# prompts when available so there's one source of truth.
try:
    from prompts import DEFAULT_NUM_IDEAS as DEFAULT_K  # type: ignore
except Exception:  # pragma: no cover
    DEFAULT_K = 8

# The six valid grounding types (plus the implicit "ungrounded"). Kept here so
# the parser can normalize/validate; the rubric modules import this set.
GROUNDING_TYPES = (
    "in_domain_extension",
    "cross_domain_transfer",
    "cross_document_synthesis",
    "contradiction",
    "scope_extension",
    "reframing",
)

# Grab the first balanced JSON array out of a chatty reply (the answer region
# should already be just the list, but reasoning models sometimes wrap it).
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class Grounding:
    source: str
    supports: str


@dataclass(frozen=True)
class Idea:
    motivation: str
    problem: str
    insight_claim: str
    grounding: Tuple[Grounding, ...] = field(default_factory=tuple)
    grounding_type: Optional[str] = None      # None / unknown == ungrounded
    raw: Optional[Dict[str, Any]] = None       # original dict, for debugging

    @property
    def is_ungrounded(self) -> bool:
        return self.grounding_type not in GROUNDING_TYPES

    def card_text(self) -> str:
        """Render the {problem, insight, motivation} card the rubric judges use."""
        return (
            f"Problem: {self.problem.strip()}\n"
            f"Insight: {self.insight_claim.strip()}\n"
            f"Motivation: {self.motivation.strip()}"
        )

    def grounding_text(self) -> str:
        """Render the declared grounding list as a readable block."""
        if not self.grounding:
            return "(no grounding provided)"
        lines = []
        for i, g in enumerate(self.grounding):
            lines.append(f"  [{i}] source: {g.source}\n      supports: {g.supports}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _coerce_grounding(value: Any) -> Tuple[Grounding, ...]:
    """Normalize the insight.grounding list into a tuple of Grounding."""
    if not isinstance(value, list):
        return tuple()
    out: List[Grounding] = []
    for item in value:
        if isinstance(item, dict):
            src = str(item.get("source", "") or "").strip()
            sup = str(item.get("supports", "") or "").strip()
            out.append(Grounding(source=src, supports=sup))
        elif item is not None:
            # tolerate a bare string citation
            out.append(Grounding(source=str(item).strip(), supports=""))
    return tuple(out)


def _idea_from_obj(obj: Any) -> Optional[Idea]:
    """Build an Idea from one list element, or None if it's structurally invalid.

    Required (non-empty str): motivation, problem, and insight.claim. The
    insight may be a nested dict (canonical schema) OR a bare string (lenient).
    """
    if not isinstance(obj, dict):
        return None
    motivation = str(obj.get("motivation", "") or "").strip()
    problem = str(obj.get("problem", "") or "").strip()

    insight = obj.get("insight")
    if isinstance(insight, dict):
        claim = str(insight.get("claim", "") or "").strip()
        grounding = _coerce_grounding(insight.get("grounding"))
        gtype = insight.get("grounding_type")
        gtype = str(gtype).strip() if gtype else None
    else:
        # lenient: insight given as a bare string, no structured grounding
        claim = str(insight or "").strip()
        grounding = tuple()
        gtype = None

    if not motivation or not problem or not claim:
        return None
    return Idea(
        motivation=motivation,
        problem=problem,
        insight_claim=claim,
        grounding=grounding,
        grounding_type=gtype,
        raw=obj,
    )


def parse_ideas(solution_str: str, *, k: Optional[int] = DEFAULT_K) -> List[Idea]:
    """Parse a policy generation into a list of structured ideas.

    Pulls the <answer> region (via inference.parsing.extract_answer_str), loads
    the JSON list, and builds one Idea per valid element. Structurally invalid
    elements are skipped (not fatal). Truncates to the first `k` ideas when `k`
    is set (emission order preserved); pass k=None to keep all.

    Returns [] if there is no parseable list at all (the caller decides whether
    that's a schema-gate failure).
    """
    answer = extract_answer_str(solution_str or "")
    if not answer:
        return []
    data = _loads_list(answer)
    if data is None:
        return []
    ideas: List[Idea] = []
    for elem in data:
        idea = _idea_from_obj(elem)
        if idea is not None:
            ideas.append(idea)
    if k is not None:
        ideas = ideas[:k]
    return ideas


def _loads_list(answer: str) -> Optional[List[Any]]:
    """json.loads the answer as a list; tolerate code fences / surrounding prose
    by grabbing the first balanced [...] span."""
    answer = answer.strip()
    # strip a leading ```json / ``` fence if present
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\s*", "", answer)
        answer = re.sub(r"\s*```$", "", answer)
    try:
        obj = json.loads(answer)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = _JSON_ARRAY_RE.search(answer)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    # Strict decoding failed (or found no clean list). Two independent recovery
    # paths, each strong where the other is weak -- take BEST-OF (most idea-shaped
    # objects), mirroring inference/parsing.py::load_json_payload:
    #   (a) object-stream scan -- some models satisfy the ideation_v0 prompt
    #       semantically (K well-formed idea objects, right keys) but NOT
    #       syntactically: a bare CONCATENATED stream `{...}\n{...}\n{...}`, no []
    #       and no commas. qwen3-8b does this on ~97% of samples; json.loads rejects
    #       it and json_repair alone collapses it to ONE object -- what silently
    #       dropped 66% of qwen's ideas at eval time. The balanced-span scanner
    #       recovers the full stream.
    #   (b) json_repair -- fixes the one-broken-object-mid-array case, where the
    #       stream scan stops at the broken object but repair reconstructs the list.
    # Short-circuiting on either loses the other's wins (the stream scan gave 4 vs
    # repair's 5 on 11 malformed-array sft-think samples). Trained checkpoints emit
    # a clean array and return above, so their already-scored numbers are unaffected.
    stream = _top_level_idea_objects(answer, _IDEA_KEYS)
    repaired = _repair_list(answer) or []
    def _n(lst):
        return sum(1 for it in (lst or [])
                   if isinstance(it, dict) and (_IDEA_KEYS & set(it.keys())))
    if _n(stream) >= _n(repaired):
        return stream or None
    return repaired or None


def _repair_list(answer: str) -> Optional[List[Any]]:
    """json_repair fallback: repaired list of idea-shaped dicts, or None if
    json_repair is unavailable / nothing idea-shaped was recovered. Mirrors
    inference/parsing.py::_repair_json_payload, normalised to always return a
    list (a lone repaired object is wrapped) so parse_ideas' loop is uniform."""
    if _repair_json is None:
        return None
    try:
        fixed = _repair_json(answer, return_objects=True)
    except Exception:
        return None
    if isinstance(fixed, dict):
        fixed = [fixed]
    if not isinstance(fixed, list):
        return None
    ideas = [it for it in fixed
             if isinstance(it, dict) and (_IDEA_KEYS & set(it.keys()))]
    return ideas or None


# --------------------------------------------------------------------------- #
# Adapters: Idea -> the dataset-instance shapes the rubric judges expect
# --------------------------------------------------------------------------- #
def idea_to_eval_card(idea: Idea, *, idea_id: Optional[str] = None,
                      title: Optional[str] = None) -> Dict[str, Any]:
    """Map an Idea onto the structured-summary 'card' an eval instance's `idea`
    field carries (problem/insight/motivation), matching what the reference
    judges render via their `_card_block`."""
    return {
        "id": idea_id,
        "title": title or "(research idea)",
        "summary": {
            "problem": idea.problem,
            "insight": idea.insight_claim,
            "motivation": idea.motivation,
        },
    }


def idea_to_reference_like(idea: Idea, *, idea_id: Optional[str] = None,
                           title: Optional[str] = None) -> Dict[str, Any]:
    """Map an Idea onto a document/reference-like shape ({title, abstract}) for
    judges that consume a free-text abstract instead of a structured card."""
    return {
        "id": idea_id,
        "title": title or "(research idea)",
        "abstract": idea.card_text(),
    }
