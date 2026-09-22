"""Parse the policy's `<answer>` block into structured SI (v1) ideas.

This is the v1 counterpart of `common/ideas.py`. The SI ideation policy
(`prompts/ideation.py::IDEATION_SI_TEMPLATE`) emits a JSON *list* of K cards with
FOUR flat string fields instead of v0's nested-insight shape:

    [
      {
        "problem_gap":  "[<opportunity pattern>] ...",
        "core_claim":   "...",
        "method":       "[<method paradigm>] ...",
        "inspiration":  "... Document <N> did X but stopped short of ..."
      },
      ... (K of them)
    ]

The two schemas line up 3+1 on the SAME slot semantics, which is why this module
is a thin adapter rather than a fork:

    slot         v0                              v1 (SI)        judged?
    the gap      problem                         problem_gap    yes
    the claim    insight.claim                   core_claim     yes
    the third    motivation                      method         yes
    provenance   insight.grounding[]/_type       inspiration    NO -- gated only

So `SIIdea` SUBCLASSES `common.ideas.Idea` and populates the v0 slots from the SI
fields, carrying `inspiration` on a new attribute. Everything downstream that
reads `.problem` / `.insight_claim` / `.motivation` or calls `.card_text()` keeps
working unchanged -- in particular the v6 reward's `_card_concat`, `_impact`,
`_distance`, its sampling seed, and its per-idea log line.

`card_text()` is overridden to render the SI labels and to EXCLUDE `inspiration`,
mirroring v0's card (which never renders grounding). That exclusion is deliberate
and load-bearing: provenance is required by the format gate but invisible to the
impact judge and untaxed by the length penalty, so the policy cannot be rewarded
for dropping it (v4's failure mode) nor for padding it.

Why a separate parser is needed at all: `common.ideas._idea_from_obj` REJECTS any
object without non-empty motivation + problem + insight.claim, so every SI card
would parse to None -> `parse_ideas` returns [] -> terminal 0.0 with no judge
calls. And `_repair_list` filters json_repair's output by the v0 key set, which
would discard every repaired SI card. Both are inside functions we otherwise want
to keep, so they are re-implemented here; the answer-region extraction and the
strict/fenced/`[...]`-span JSON loading are IMPORTED from `common.ideas`, not
copied.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Same path bootstrap as common/ideas.py so `prompts` resolves whether this is
# imported by verl via custom_reward_function.path, by a test, or as a script.
_GRPO_ROOT = Path(__file__).resolve().parents[1]      # .../grpo_training
_REPO_ROOT = _GRPO_ROOT.parent                        # repo root (has prompts/)
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.ideas import (  # noqa: E402
    DEFAULT_K,
    Idea,
    _loads_list,
    _top_level_idea_objects,
    extract_answer_str,
)

# json_repair fallback, same contract as common/ideas.py: absent -> skipped.
try:
    from json_repair import repair_json as _repair_json  # type: ignore
except Exception:  # pragma: no cover
    _repair_json = None

# Keys that mark a dict as an SI idea (mirrors inference/parsing.py::_SI_IDEA_KEYS).
SI_IDEA_KEYS = {"problem_gap", "core_claim", "method", "inspiration"}

# --------------------------------------------------------------------------- #
# Closed label vocabularies (the two taxonomies the SI prompt teaches).
# Kept verbatim from prompts/ideation.py's TAXONOMY block; the format gate
# validates against these the way v6 validates v0's grounding_type against
# GROUNDING_TYPES. `[Other]` is the prompt's explicit escape hatch.
# --------------------------------------------------------------------------- #
OPPORTUNITY_PATTERNS = (
    "Puzzle / Contradiction",
    "Explanation Gap",
    "Scope Mismatch",
    "Evidence Gap",
    "Bridge Opportunity",
    "Failure / Risk Gap",
    "Resource Bottleneck",
)

METHOD_PARADIGMS = (
    "Synthesis / Unification",
    "Relax / Extend Scope",
    "Robustification",
    "Formal Derivation",
    "Empirical Mapping",
    "Artifact / System",
    "Optimization / Search",
)

OTHER_LABEL = "Other"

_LEADING_LABEL_RE = re.compile(r"^\s*\[([^\]]{1,60})\]")


def normalize_label(label: str) -> str:
    """Canonical form for label comparison: casefolded, whitespace collapsed, and
    spacing around '/' normalized.

    The taught labels are two-sided ("Synthesis / Unification"), but generations
    drift to "Synthesis/Unification" or "synthesis / unification" -- 15 such rows
    in train_multi. Those are the SAME label and must not fail the gate, so we
    compare normalized rather than by exact string equality."""
    s = re.sub(r"\s+", " ", str(label or "")).strip()
    s = re.sub(r"\s*/\s*", "/", s)
    return s.casefold()


def _accepted_forms(labels) -> set:
    """Normalized forms accepted for one axis: the taught label, plus the first
    segment of a two-sided label, plus [Other].

    The segment alias exists because generations truncate two-sided labels --
    "[Synthesis]" for "[Synthesis / Unification]" accounts for 15 of the 24,015
    ideas in train_multi. It is the same label, so failing the gate on it would
    penalize the policy for a wording slip the SFT data itself contains. The
    first segments are unique across both taxonomies (Synthesis, Relax, Puzzle,
    Failure, Artifact, Optimization), so this cannot alias one axis onto the
    other."""
    out = {normalize_label(OTHER_LABEL)}
    for v in labels:
        out.add(normalize_label(v))
        head = v.split("/")[0].strip()
        if head and head != v:
            out.add(normalize_label(head))
    return out


_OPPORTUNITY_NORM = _accepted_forms(OPPORTUNITY_PATTERNS)
_METHOD_NORM = _accepted_forms(METHOD_PARADIGMS)


def leading_label(value: str) -> Optional[str]:
    """The bracketed label at the very start of a field, or None if absent.

    The prompt requires the label ONLY at the start ("do not repeat it in the
    prose"), so an interior "[...]"" does not count."""
    m = _LEADING_LABEL_RE.match(str(value or ""))
    return m.group(1) if m else None


def has_valid_opportunity_label(problem_gap: str) -> bool:
    """True if `problem_gap` begins with a bracketed OPPORTUNITY PATTERN (or
    [Other]), compared normalized."""
    lab = leading_label(problem_gap)
    return lab is not None and normalize_label(lab) in _OPPORTUNITY_NORM


def has_valid_method_label(method: str) -> bool:
    """True if `method` begins with a bracketed METHOD PARADIGM (or [Other]),
    compared normalized."""
    lab = leading_label(method)
    return lab is not None and normalize_label(lab) in _METHOD_NORM


# --------------------------------------------------------------------------- #
# The SI idea
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SIIdea(Idea):
    """One SI card, stored in v0's slots so v0-shaped consumers keep working.

    problem <- problem_gap, insight_claim <- core_claim, motivation <- method.
    `grounding` / `grounding_type` stay at their empty defaults (SI has no
    structured citation list); `inspiration` carries the provenance prose."""

    inspiration: str = ""

    # Convenience aliases so SI-facing code can read the SI names.
    @property
    def problem_gap(self) -> str:
        return self.problem

    @property
    def core_claim(self) -> str:
        return self.insight_claim

    @property
    def method(self) -> str:
        return self.motivation

    def card_text(self) -> str:
        """The judged card: gap + claim + method. `inspiration` is EXCLUDED by
        construction, so it is neither scored by the impact judge nor priced by
        the length penalty -- it is enforced by the format gate alone."""
        return (
            f"Problem gap: {self.problem.strip()}\n"
            f"Core claim: {self.insight_claim.strip()}\n"
            f"Method: {self.method.strip()}"
        )

    def provenance_text(self) -> str:
        """SI analogue of Idea.grounding_text(), for logging/debugging only."""
        return self.inspiration.strip() or "(no inspiration provided)"


def si_idea_from_obj(obj: Any) -> Optional[SIIdea]:
    """Build an SIIdea from one list element, or None if structurally invalid.

    All FOUR fields must be non-empty strings: the three judged slots plus
    `inspiration`. Requiring inspiration here (and again in the format gate) is
    what stops the policy from silently dropping the untaxed, unjudged field."""
    if not isinstance(obj, dict):
        return None
    gap = str(obj.get("problem_gap", "") or "").strip()
    claim = str(obj.get("core_claim", "") or "").strip()
    method = str(obj.get("method", "") or "").strip()
    inspiration = str(obj.get("inspiration", "") or "").strip()
    if not gap or not claim or not method or not inspiration:
        return None
    return SIIdea(
        motivation=method,
        problem=gap,
        insight_claim=claim,
        inspiration=inspiration,
        raw=obj,
    )


def _si_count(lst) -> int:
    """How many list elements look like an SI card (carry an SI key)."""
    return sum(1 for it in (lst or [])
               if isinstance(it, dict) and (SI_IDEA_KEYS & set(it.keys())))


def _loads_list_si(answer: str) -> Optional[List[Any]]:
    """Strict list load with an SI-keyed object-stream scan + repair fallback.

    `common.ideas._loads_list` handles code fences and the widest balanced [...]
    span, so we try it first for the clean-array case. But its two RECOVERY paths
    (the `{...}{...}` object-stream scan and the json_repair filter) both key on
    the v0 key set, so for a bare SI object stream -- which qwen3-8b emits on
    nearly every sample: pretty-printed `{...}\n{...}\n{...}`, no [] and no commas
    -- v0's scan finds 0 objects and json_repair collapses the stream to ONE dict.
    That silently dropped ~2/3 of every SI model's ideas (12 samples -> 12 ideas
    instead of ~35). So we redo BOTH recovery paths here with the SI keys, taking
    best-of exactly like the v0 loader, and only accept _loads_list's result
    directly when it already found a clean array."""
    answer = answer.strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\s*", "", answer)
        answer = re.sub(r"\s*```$", "", answer)
    # Clean-array fast path: a real JSON list of SI cards. (A v0 stream/repair
    # result would carry no SI keys, so guard on the SI count before trusting it.)
    data = _loads_list(answer)
    if _si_count(data) > 0:
        return data
    # SI-keyed recovery: object-stream scan (best for the bare `{...}{...}` stream)
    # vs json_repair (best for one-broken-object-mid-array); take whichever yields
    # more SI-shaped objects -- same best-of policy as common.ideas._loads_list.
    stream = _top_level_idea_objects(answer, SI_IDEA_KEYS)
    repaired: List[Any] = []
    if _repair_json is not None:
        try:
            fixed = _repair_json(answer, return_objects=True)
            if isinstance(fixed, dict):
                fixed = [fixed]
            if isinstance(fixed, list):
                repaired = [it for it in fixed if isinstance(it, dict)
                            and (SI_IDEA_KEYS & set(it.keys()))]
        except Exception:
            repaired = []
    if _si_count(stream) >= _si_count(repaired):
        return stream or None
    return repaired or None


def parse_ideas_si(solution_str: str, *, k: Optional[int] = DEFAULT_K) -> List[SIIdea]:
    """Parse a policy generation into SI ideas. Drop-in for common.ideas.parse_ideas.

    Same contract: pulls the <answer> region, loads the JSON list, builds one
    SIIdea per valid element (invalid elements skipped, not fatal), truncates to
    the first `k` (pass k=None to keep all). Returns [] when nothing parses --
    the caller decides whether that's a schema-gate failure."""
    answer = extract_answer_str(solution_str or "")
    if not answer:
        return []
    data = _loads_list_si(answer)
    if data is None:
        return []
    ideas: List[SIIdea] = []
    for elem in data:
        idea = si_idea_from_obj(elem)
        if idea is not None:
            ideas.append(idea)
    if k is not None:
        ideas = ideas[:k]
    return ideas


def si_idea_from_fields(obj: Dict[str, Any]) -> Optional[SIIdea]:
    """Public alias of si_idea_from_obj for callers holding a plain dict (e.g.
    data prep rendering a gold card through the same slot renderer)."""
    return si_idea_from_obj(obj)


# The stored gold strings (`target_si_idea_grounded_str`, and `target_idea_str` in
# train_multi) are a flat "problem_gap: ...\ncore_claim: ...\nmethod: ...\n
# inspiration: ..." block. Verified exhaustively: all 8006 train_multi, all 8578
# train_gold and 2056/2058 test golds parse to exactly these four keys (the 2 are
# empty strings). Values may span multiple lines, so each field runs to the next
# label rather than to the next newline.
_SI_STR_FIELD_RE = re.compile(
    r"^(problem_gap|core_claim|method|inspiration)\s*:\s*", re.M)


def parse_si_card_str(text: str) -> Dict[str, str]:
    """Parse a flat "problem_gap: ... / core_claim: ... / method: ... /
    inspiration: ..." gold string into a dict. Missing fields are simply absent,
    so the caller can decide (render_si_card falls back)."""
    text = str(text or "")
    marks = list(_SI_STR_FIELD_RE.finditer(text))
    out: Dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1)] = text[m.end():end].strip()
    return out


def render_si_card(obj: Any) -> str:
    """Render a gold SI card -- dict OR flat labeled string -- as the 3-field
    judged card, i.e. exactly the shape SIIdea.card_text() gives a candidate.

    Used by data prep for `ground_truth`. The stored gold strings carry all FOUR
    fields including `inspiration`; the candidate card deliberately has three, so
    passing the stored string through verbatim would show the impact judge a
    4-field gold against a 3-field candidate -- a systematic format asymmetry on
    the one comparison the reward turns on. Accepts a string because the RL source
    (train_multi) ships only `target_idea_str`, never the dict. Returns "" for
    input with none of the four fields, so callers can detect an unusable gold."""
    if isinstance(obj, str):
        obj = parse_si_card_str(obj)
    if not isinstance(obj, dict):
        return ""
    idea = si_idea_from_obj(obj)
    if idea is not None:
        return idea.card_text()
    gap = str(obj.get("problem_gap", "") or "").strip()
    claim = str(obj.get("core_claim", "") or "").strip()
    method = str(obj.get("method", "") or "").strip()
    if not (gap or claim or method):
        return ""
    return (f"Problem gap: {gap}\n"
            f"Core claim: {claim}\n"
            f"Method: {method}")
