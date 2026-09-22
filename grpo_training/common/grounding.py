"""Grounding-fidelity rubric: is an idea's insight genuinely grounded in the
cited source documents, or is the citation decorative / fabricated?

NET-NEW (no Mac reference to port). Designed from thinking/reward_formulations.md
(the grounding_type taxonomy + the "load-bearing" check) and the prompts/ideation.py
schema (insight.grounding = [{source, supports}], insight.grounding_type).

Like common.novelty, this is TRANSPORT-AGNOSTIC: rubric text + prompt builder +
parser + scoring math only; the client lives in the caller.

The judge sees the insight CLAIM, its DECLARED grounding list, the declared
grounding_type, AND the full source documents -- so it can verify the cited
source actually exists (catch fabrication) and is load-bearing (catch decorative
citations). We score four 1-5 features:

  source_exists     -- does the cited source actually appear in the documents?
  support_faithful  -- does the cited source actually establish what `supports` claims?
  load_bearing      -- would the insight collapse if the cited docs were removed?
  type_correct      -- does the insight match its declared grounding_type?

Eval reports the mean `fidelity` in [1,5]. The F9-B reward needs a binary gate
E in {0,1}: `features_to_E` returns 1 iff grounding_type is non-ungrounded AND
the load-bearing-centric features clear a threshold. One rubric, two reductions
(mirrors novelty's max-overlap vs per-field-N split).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

RUBRIC_VERSION = "grounding-fidelity-v1"

# The six valid grounding types (plus the implicit "ungrounded"). Imported from
# common.ideas where possible so there's one source of truth.
try:
    from common.ideas import GROUNDING_TYPES  # type: ignore
except Exception:  # pragma: no cover
    GROUNDING_TYPES = (
        "in_domain_extension", "cross_domain_transfer", "cross_document_synthesis",
        "contradiction", "scope_extension", "reframing",
    )

GROUNDING_FEATURE_KEYS = ["source_exists", "support_faithful",
                          "load_bearing", "type_correct"]

# Features that gate E (the reward's binary grounding pass). load_bearing is the
# crux per the reward doc ("if removing the cited docs leaves the insight intact,
# E=0"); source_exists guards against fabricated citations.
_E_GATING_FEATURES = ["source_exists", "load_bearing"]
DEFAULT_E_THRESHOLD = 3  # gating features must be >= this (on the 1-5 scale)


SYSTEM_MESSAGE = (
    "You are an expert research scientist auditing whether a research idea's "
    "central insight is genuinely GROUNDED in the source documents it cites. "
    "You are given the insight, the documents it claims to build on, and the "
    "specific grounding the author declared. Judge fidelity of grounding -- not "
    "novelty, quality, or impact. Be skeptical of citations that are decorative "
    "(the insight would survive deleting them) or fabricated (the cited source "
    "does not actually appear, or does not say what is claimed). Return only a "
    "single JSON object."
)

TEMPLATE = """Below is a research INSIGHT, the GROUNDING the author declared for it (one or
more source references with what each is said to support),{type_intro}{docs_intro}

Judge how faithfully the insight is grounded in the cited sources. Score {n_features}
features, each 1-5. USE THE FULL SCALE.

  source_exists -- do the cited source(s) actually appear among the SOURCE
    DOCUMENTS below?
      1 = fabricated: the cited source is not present at all
      2 = a vaguely matching document exists but the citation is wrong/mismatched
      3 = a plausibly matching document exists but identification is loose
      4 = the cited source clearly maps to a real document, minor ambiguity
      5 = exact, unambiguous match to a provided document

  support_faithful -- does the cited source actually ESTABLISH what the
    declared `supports` text claims it does?
      1 = misrepresents the source; the source does not say this
      2 = weak/strained connection to what the source actually shows
      3 = partially supported; the source backs some but not all of the claim
      4 = well supported with a minor gap or overstatement
      5 = the source clearly and fully establishes the claimed support

  load_bearing -- would the insight COLLAPSE if the cited documents were
    removed? (decorative citation test)
      1 = decorative: the insight stands completely without the cited docs
      2 = the docs add color but the insight does not depend on them
      3 = the docs support the insight but it could be re-grounded elsewhere
      4 = the insight largely depends on the specific cited findings
      5 = the insight strictly requires the cited docs; remove them and it falls apart

{type_rubric}=== INSIGHT ===
{insight_claim}
{type_block}
=== DECLARED GROUNDING ===
{grounding_block}

=== SOURCE DOCUMENTS ===
{documents_block}

Return ONLY a JSON object of the form:
{{
  "source_exists": 1 | 2 | 3 | 4 | 5,
  "support_faithful": 1 | 2 | 3 | 4 | 5,
  "load_bearing": 1 | 2 | 3 | 4 | 5,{type_json}
  "rationale": "1-3 sentences citing which source supports (or fails to support) the insight."
}}
Do not output anything outside the JSON object."""

# The `type_correct` feature, spliced in only when the schema HAS a grounding_type.
# Kept as separate fragments rather than a "(none declared)" placeholder because
# telling the judge the type is "ungrounded" primes it to score the whole card as
# decorative -- see build_messages.
_TYPE_INTRO = " the declared\nGROUNDING TYPE,"
# Two intro tails so BOTH variants wrap like hand-written prose. With the type
# clause the sentence continues on line 2; without it, line 1 has room. The v0
# rendering stays byte-identical to the pre-configurable template.
_DOCS_INTRO_WITH_TYPE = " and the full SOURCE DOCUMENTS available to the author."
_DOCS_INTRO_NO_TYPE = " and the full\nSOURCE DOCUMENTS available to the author."

# {type_rubric} is spliced in with NO separating newline of its own, so this fragment
# owns its own trailing blank line. That way removing it leaves exactly one blank line
# before === INSIGHT === (not three), and keeping it reproduces the original spacing
# byte-for-byte.
_TYPE_RUBRIC = """  type_correct -- does the insight actually match its declared GROUNDING TYPE?
      1 = clear mismatch (e.g. labeled cross_domain_transfer but stays in-domain)
      2 = mostly mismatched
      3 = partially fits the declared type
      4 = fits the declared type with a minor stretch
      5 = exactly fits the declared type

"""

_TYPE_JSON = '\n  "type_correct": 1 | 2 | 3 | 4 | 5,'


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def documents_block(documents: List[str]) -> str:
    """Render the population of source documents (raw strings) with indices.

    1-INDEXED, to match `prompts/ideation.py::format_documents` -- the SAME
    numbering shown to the model that wrote the insight being judged (both at
    generation time and in `training/sft/data_generation/formats/*.py`'s
    backward-trace prompts, which also instruct "Document <N>" citations).
    This used to be 0-indexed here while generation is 1-indexed, so every
    "Document N" citation the model wrote landed on the judge's "DOCUMENT N",
    which is actually the (N+1)-th document -- a silent off-by-one that made
    every citation check the wrong (or, for the last document, a nonexistent)
    source. See thinking_design_v2 grounding investigation: judge rationales
    referencing "Document 0" (a label the model was never shown) and "Document
    N does not exist" for the highest-numbered citation were the tell."""
    if not documents:
        return "(no source documents provided)"
    return "\n\n".join(f"--- DOCUMENT {i} ---\n{str(d).strip()}"
                       for i, d in enumerate(documents, start=1))


def build_messages(insight_claim: str, grounding_text: str, grounding_type: Optional[str],
                   documents: List[str], *, ask_type: Optional[bool] = None
                   ) -> List[Dict[str, str]]:
    """Render the rubric prompt.

    `ask_type` controls whether the `type_correct` feature is asked for at all.
    Default (None) = ask iff a grounding_type was actually declared.

    WHY IT IS OMITTED RATHER THAN BLANK. `grounding_type` is a v0-schema field
    (prompts/ideation.py's insight.grounding_type, one of six taxonomy values). The
    SI schema has no equivalent -- SIIdea leaves it empty by construction. Rendering
    it as "(none declared / ungrounded)" told the judge, in the same prompt where it
    is asked whether the citation is decorative, that the idea is UNGROUNDED. That is
    a leading premise, not neutral missing data: it biases `load_bearing` -- the one
    feature the reward actually prices -- downward on every SI card, and it invites a
    score for a feature the card was never asked to have. So for a schema without
    types we drop the intro clause, the rubric entry, the DECLARED GROUNDING TYPE
    section, and the JSON key, and the judge scores the three features that apply.

    Callers that pass a type get byte-identical output to before this change."""
    if ask_type is None:
        ask_type = grounding_type is not None and str(grounding_type).strip() != ""
    prompt = TEMPLATE.format(
        n_features="four" if ask_type else "three",
        type_intro=_TYPE_INTRO if ask_type else "",
        docs_intro=_DOCS_INTRO_WITH_TYPE if ask_type else _DOCS_INTRO_NO_TYPE,
        type_rubric=_TYPE_RUBRIC if ask_type else "",
        insight_claim=(insight_claim or "").strip() or "(no insight provided)",
        # An ABSENT type on a schema that HAS the field is real signal (the v0 card
        # failed to declare one) and keeps its original wording -- unchanged from
        # before this became configurable.
        type_block=("\n=== DECLARED GROUNDING TYPE ===\n"
                    f"{grounding_type or '(none declared / ungrounded)'}\n"
                    if ask_type else ""),
        type_json=_TYPE_JSON if ask_type else "",
        grounding_block=grounding_text or "(no grounding provided)",
        documents_block=documents_block(documents),
    )
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]


# --------------------------------------------------------------------------- #
# Parsing (shares the extraction style with common.novelty)
# --------------------------------------------------------------------------- #
def extract_json(raw: str) -> Optional[Any]:
    """Last-balanced-{...} extraction (handles chatty/reasoning replies)."""
    if not raw:
        return None
    candidates: List[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(raw[start:i + 1])
    parsed: Optional[Any] = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
    return parsed


def clamp_score(v: Any) -> Optional[int]:
    try:
        s = int(v)
    except Exception:
        return None
    return s if 1 <= s <= 5 else None


def parse_features(obj: Dict[str, Any], *, require_type: bool = True
                   ) -> Optional[Dict[str, Any]]:
    """Validate the 1-5 grounding features (+ rationale). None if any REQUIRED
    feature is missing / out of range -- that strictness is what makes a truncated
    reply a judge failure (-> imputed) rather than a silently partial score.

    `require_type=False` for schemas with no grounding_type, where build_messages
    never asked for `type_correct`: its absence is then correct, not a failure. It is
    still accepted if the judge volunteers it."""
    if not isinstance(obj, dict):
        return None
    required = list(GROUNDING_FEATURE_KEYS)
    if not require_type:
        required = [k for k in required if k != "type_correct"]
    feats: Dict[str, Any] = {}
    for k in required:
        s = clamp_score(obj.get(k))
        if s is None:
            return None
        feats[k] = s
    if not require_type:
        extra = clamp_score(obj.get("type_correct"))
        if extra is not None:
            feats["type_correct"] = extra
    feats["rationale"] = str(obj.get("rationale") or "").strip()
    return feats


def fidelity(feats: Dict[str, Any]) -> float:
    """Grounding fidelity = mean of the scored features, in [1, 5].

    Averages over the features PRESENT, so a three-feature (no grounding_type)
    result is on the same 1-5 scale as a four-feature one rather than being
    dragged toward 0 by a missing key."""
    vals = [feats[k] for k in GROUNDING_FEATURE_KEYS if isinstance(feats.get(k), (int, float))]
    return (sum(vals) / len(vals)) if vals else 0.0


def overall_score(feats: Optional[Dict[str, Any]], grounding_type: Optional[str]
                  ) -> Optional[float]:
    """Single graded grounding score in [0, 1] -- the headline number to compare
    ideas / models by (vs. the binary E gate, which is for the reward).

    Rescales fidelity (1-5) to [0,1] as (mean-1)/4, then HARD-ZEROS an ungrounded
    idea: if the declared grounding_type is not a valid type, the insight isn't
    grounded at all, so overall = 0 regardless of feature scores. Returns None if
    the idea was unparseable (no features)."""
    if feats is None:
        return None
    if grounding_type not in GROUNDING_TYPES:
        return 0.0
    return (fidelity(feats) - 1.0) / 4.0


def aggregate(feats: Optional[Dict[str, Any]], grounding_type: Optional[str], *,
              threshold: int = DEFAULT_E_THRESHOLD) -> Dict[str, Any]:
    """Instance-level grounding result: graded `overall` [0,1] (headline) +
    `fidelity` (1-5 raw mean) + binary `E` gate (for the F9-B reward)."""
    if feats is None:
        return {"overall": None, "fidelity": None, "E": 0,
                "grounding_type": grounding_type, "features": None}
    return {
        "overall": overall_score(feats, grounding_type),
        "fidelity": fidelity(feats),
        "E": features_to_E(feats, grounding_type, threshold=threshold),
        "grounding_type": grounding_type,
        "features": feats,
    }


# --------------------------------------------------------------------------- #
# F9-B reconciliation: binary E gate
# --------------------------------------------------------------------------- #
def features_to_E(feats: Dict[str, Any], grounding_type: Optional[str], *,
                  threshold: int = DEFAULT_E_THRESHOLD) -> int:
    """E in {0,1}: 1 iff the grounding_type is a valid (non-ungrounded) type AND
    the load-bearing-centric gating features clear `threshold`. Catches both
    fabricated citations (low source_exists) and decorative ones (low
    load_bearing)."""
    if grounding_type not in GROUNDING_TYPES:
        return 0
    if not feats:
        return 0
    for k in _E_GATING_FEATURES:
        v = feats.get(k)
        if not isinstance(v, (int, float)) or v < threshold:
            return 0
    return 1


# --------------------------------------------------------------------------- #
# F9-B reconciliation: GRADED g multiplier (the reward's actual gate)
# --------------------------------------------------------------------------- #
# Measured on 368 judged ideas from job 328 (extras/validate_grounding_proxy.py),
# 168 from steps 7-22 and 200 from steps 482-507:
#
#                     early   late    corr(lexical off-topic proxy), late
#   load_bearing        2.99   1.76        +0.401
#   type_correct        4.02   3.04        +0.271
#   support_faithful    1.97   1.56        +0.197
#   source_exists       2.45   2.38        -0.045
#
# WHY load_bearing ALONE and not `fidelity` or the E gate:
#   * It is the only feature that tracks the observed failure. `source_exists` is
#     FLAT and uncorrelated -- the drifted policy cites REAL findings CORRECTLY and
#     then bridges to an unrelated idea ("analogous to ..."). Averaging a null
#     feature into the gate dilutes the signal ~4x (fidelity's early->late drop is
#     23.5% vs load_bearing's 41.0%).
#   * Folding in `support_faithful` also HURT (drop 41.0% -> 33.0% mean / 25.8%
#     min): it already starts near the floor (1.97) so it has little room to
#     signal, and it is +0.51 correlated with load_bearing late anyway.
#   * features_to_E passes only 9.5% of late rollouts at threshold 3. Zeroing >90%
#     of contributions collapses `body` and destroys the within-group advantage
#     variance GRPO needs -- the best-of-n starvation that made rollout_n=4 fail.
#
# The shape is GRADED (not binary): features_to_E passes only 9.5% of late
# rollouts, and zeroing >90% of contributions is the best-of-n starvation that made
# rollout_n=4 fail to learn.
#
# DEFAULTS ARE THE RUBRIC'S OWN MIN-MAX: lo=1, hi=5, floor=0, i.e. a plain
# normalisation g = (load_bearing - 1) / 4 -> 1/2/3/4/5 = 0.00/0.25/0.50/0.75/1.00.
# The scale the judge actually scores on maps straight onto the multiplier with no
# extra calibration layer, so `g` is readable directly off `load_bearing_mean` and
# there is no tuned constant to justify.
#
# `lo`/`hi`/`floor` remain overridable because the clip they drive is what makes the
# gate SATURATE and FLOOR, and both were measured to matter on 328's 368 ideas:
#
#   config              early g  late g  late sd   note
#   lo=1 hi=5 floor=0     0.50    0.19    ~0.24    default: linear, no clip active
#   lo=1 hi=3 floor=0.25  0.79    0.47     0.304   saturating+floored variant
#
# What the non-default variant buys, if the default proves too harsh in the smoke
# run: hi=3 stops paying for load_bearing 3->4->5 (level 3 is "supports the insight
# but could be re-grounded elsewhere", which is where SFT-init already sits at 2.99),
# so there is no gradient toward keyword-stuffing the documents or toward merely
# ASSERTING dependence; and floor>0 keeps the 57.5% of late ideas at load_bearing=1
# off an identical 0.0, where they share no gradient and earn zero diversity credit.
LB_LO = float(os.environ.get("GROUNDING_LB_LO", "1.0"))      # g == floor at/below this
LB_HI = float(os.environ.get("GROUNDING_LB_HI", "5.0"))      # g == 1.0 at/above this
LB_FLOOR = float(os.environ.get("GROUNDING_LB_FLOOR", "0.0"))  # min multiplier


def features_to_g(feats: Optional[Dict[str, Any]], grounding_type: Optional[str], *,
                  lo: float = LB_LO, hi: float = LB_HI,
                  floor: float = LB_FLOOR, require_type: bool = True) -> float:
    """Graded grounding multiplier in [floor, 1.0], from load_bearing.

    Default mapping -- the rubric's own 1-5 min-max, (load_bearing - 1) / 4:
        load_bearing  1 -> 0.00   2 -> 0.25   3 -> 0.50   4 -> 0.75   5 -> 1.00

    At the defaults (lo=1, hi=5, floor=0) the clip is INERT: the ramp already lands
    in [0, 1]. It only bites under a saturating/floored config such as
    lo=1 hi=3 floor=0.25 -- see the comment block above for what that variant buys.

    `feats is None` means the JUDGE FAILED, which is MISSING DATA, not evidence of
    decorative citation -- it returns `floor` here, but the reward MUST NOT use that
    value: it imputes to the trailing-window mean instead so an outage is a GRPO
    no-op rather than a grounding penalty (see f9b_impdist_len_v6._impute_g).

    An invalid/absent grounding_type IS a real signal (the idea declared no valid
    grounding at all), so it returns `floor` and the reward uses it. At the default
    floor=0 that ZEROES the idea's whole contribution -- deliberate, and reachable
    only for a card that passed the format gate yet named no valid grounding_type.

    `require_type=False` disables that check, for schemas that have no grounding_type
    field at all -- SI cards carry free-text `inspiration` and leave grounding_type
    empty by construction, which would otherwise pin every SI idea to `floor`."""
    if require_type and grounding_type not in GROUNDING_TYPES:
        return floor
    if not feats:
        return floor
    v = feats.get("load_bearing")
    if not isinstance(v, (int, float)):
        return floor
    if hi <= lo:  # degenerate config -> binary at lo
        return 1.0 if v >= hi else floor
    return max(floor, min(1.0, (float(v) - lo) / (hi - lo)))
