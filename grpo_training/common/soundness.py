"""Shared soundness-judge plumbing (single source of truth for the interface).

The soundness judge rates the scientific rigor of a research idea. We validated
it on SoundnessBench (1,099 ML proposals, binary rigor_bucket ground truth) in
reward/soundness_eval/SoundnessBench: the strongest affordable configuration was
the *scale4* judge (a 1-4 soundness rating) sampled with self-consistency votes,
run on a self-hosted gpt-oss-120b. Claude is a better judge but too expensive
in-loop.

This module carries the scale4 SYSTEM/USER prompt, the 1-4 rating parser, and the
decode params -- ported VERBATIM from
``reward/soundness_eval/SoundnessBench/rigorbench/evaluation/{prompt,run}.py`` so
the training reward does NOT depend on the external benchmark repo being on the
path. The prompt wording must stay byte-identical to what was benchmarked; any
change invalidates the balanced-accuracy numbers in the memo. Keep it here and
nowhere else (mirrors common/scijudge.py).

Reward-relevant findings baked into the defaults (see project memory):
  * The 1-4 scale collapses to a de-facto 2-vs-3 boundary (rungs 1 & 4 starve),
    so votes DENOISE the boundary rather than resolve four levels -- N=5 votes at
    a non-zero temperature is the validated config.
  * gpt-oss is the affordable judge; claude-* is a ceiling reference only.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from common import gateway

DEFAULT_MODEL = os.environ.get("SOUNDNESS_MODEL", "openai/gpt-oss-120b")

# --------------------------------------------------------------------------- #
# Prompt (scale4) -- ported verbatim from rigorbench/evaluation/prompt.py.
# The judge sees a hypothesis + experiment PLAN (no run results); the rungs
# assess the soundness of the design, not the absence of executed experiments.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert ML researcher and peer reviewer. Rate the scientific "
    "soundness of a research idea on a 1-4 scale (and your assessment confidence "
    "from 1 to 5) from its hypothesis and experiment plan.\n\n"
    "You are reviewing a PROPOSAL, not a completed paper: no run results are "
    "provided. Judge the soundness of the design and reasoning, not the absence "
    "of executed experiments.\n\n"
    "Output the assessment as a JSON object, including a detailed step-by-step "
    "justification for the rating selected."
)

_SCALE4_RUBRIC = """Rate the scientific soundness of this research idea on a 1-4 scale. The judgement is about the proposal's design and reasoning -- you are given a hypothesis and an experiment PLAN, not executed results, so do not penalize the mere absence of run results; judge whether the plan, if carried out as described, would soundly establish the claim.
- 4 (Excellent): The idea is technically flawless and theoretically sound; the proposed experiments are rigorous and well controlled, with appropriate baselines and metrics that would decisively test the hypothesis.
- 3 (Good): Generally sound and technically correct. The hypothesis is meaningful and the plan would credibly test it, though there may be minor flaws or slight limitations.
- 2 (Fair): Noticeable weaknesses in the methodology, experimental design, or theoretical reasoning -- e.g. uncontrolled confounds, a metric that does not clearly measure the claim, or conclusions broader than the plan can support -- but the core idea holds some value.
- 1 (Poor): Major technical flaws, incorrect reasoning/proofs, or a design that could not establish the central claim even if executed."""

USER_TEMPLATE = """{rubric}

Confidence Score Scale:
- 1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.
- 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
- 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
- 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

HYPOTHESIS:
{hypothesis}

EXPERIMENT:
{experiment}

Output format:
{{
  "justification": "<Think step-by-step, provide detailed justification>",
  "soundness_rating": <1, 2, 3, or 4>,
  "confidence": <1-5 integer>
}}

Constraints:
- soundness_rating must be an integer in [1, 4]
- confidence must be an integer in [1, 5]
"""

# The reward feeds the raw generated idea into the HYPOTHESIS slot; there is no
# separate experiment plan yet (the policy does not emit a protocol), so the
# EXPERIMENT slot gets this sentinel -- matching how rigorbench renders an empty
# slot ("(none)") so the prompt shape is unchanged.
NO_EXPERIMENT = "(none)"


# SI-field render mapping -- IDENTICAL to the validated pipeline
# (reward/soundness_eval/SoundnessBench/scripts/si_judge_eval.py::
# render_si_to_judge_slots), so the balanced-accuracy numbers from the memo
# transfer to the reward: the judge sees the same two slots it was benchmarked on.
#   HYPOTHESIS  <- problem_gap + core_claim   (the claim being made)
#   EXPERIMENT  <- method + inspiration       (how it's done + what it's grounded in)
# The generation will carry the SI fields directly; until it does, we fall back
# to today's ideation schema (problem/insight/grounding) so the reward still runs.
_SI_HYP_FIELDS = ("problem_gap", "core_claim")
_SI_EXP_FIELDS = ("method", "inspiration")


def _fmt_inspiration(insp) -> str:
    """Render an inspiration/grounding value (list of {source, how_used} or
    {source, supports}, or a bare string) into readable lines."""
    if isinstance(insp, list):
        lines = []
        for i in insp:
            if isinstance(i, dict):
                src = i.get("source", "?")
                how = i.get("how_used") or i.get("supports") or ""
                lines.append(f"  - {src}: {how}".rstrip(": "))
            elif i is not None:
                lines.append(f"  - {i}")
        return "\n".join(lines)
    return str(insp or "").strip()


def render_idea_to_slots(idea: dict) -> tuple[str, str]:
    """Map a generated idea dict onto the judge's (HYPOTHESIS, EXPERIMENT) slots.

    Prefers the SI fields (problem_gap/core_claim/method/inspiration) using the
    validated mapping. Falls back to the current ideation schema
    (problem/insight[.claim]/grounding) when SI fields are absent, and to the
    ``(none)`` sentinel for a slot with no content (e.g. no method/plan yet).
    """
    idea = idea or {}

    gap = str(idea.get("problem_gap") or idea.get("problem") or "").strip()
    claim = idea.get("core_claim")
    if not claim:
        insight = idea.get("insight")
        claim = insight.get("claim") if isinstance(insight, dict) else insight
    claim = str(claim or "").strip()

    method = str(idea.get("method") or "").strip()
    insp = idea.get("inspiration")
    if insp is None:
        insight = idea.get("insight")
        insp = insight.get("grounding") if isinstance(insight, dict) else idea.get("grounding")
    insp_block = _fmt_inspiration(insp)

    hyp_parts = []
    if gap:
        hyp_parts.append(f"Problem / gap: {gap}")
    if claim:
        hyp_parts.append(f"Core claim: {claim}")
    hypothesis = "\n\n".join(hyp_parts)

    exp_parts = []
    if method:
        exp_parts.append(f"Method: {method}")
    if insp_block:
        exp_parts.append(f"Grounding / inspirations:\n{insp_block}")
    experiment = "\n\n".join(exp_parts)

    return hypothesis or NO_EXPERIMENT, experiment or NO_EXPERIMENT


def build_messages(hypothesis: str, experiment: str = NO_EXPERIMENT) -> list[dict]:
    """Chat messages asking the judge to rate one idea's soundness (1-4).

    ``hypothesis`` carries the generated idea; ``experiment`` is the plan slot,
    which defaults to the ``(none)`` sentinel because the policy emits no protocol.
    """
    user = USER_TEMPLATE.format(
        rubric=_SCALE4_RUBRIC,
        hypothesis=(hypothesis or "").strip() or NO_EXPERIMENT,
        experiment=(experiment or "").strip() or NO_EXPERIMENT,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Parser -- ported from rigorbench/evaluation/run.py (_extract_first_json_object
# + _rating_to_bucket). We only need the 1-4 rating for the reward.
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    return _FENCE_RE.sub("", text or "")


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in ``text`` (fences stripped)."""
    text = _strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def parse_rating(text: str) -> Optional[int]:
    """Return the 1-4 ``soundness_rating`` from a judge response, or None.

    Tries strict JSON first, then a regex fallback for a malformed body (mirrors
    rigorbench's tolerance: a truncated/fenced object still yields the rating).
    Clamps to [1, 4]; non-numeric -> None.
    """
    if not text:
        return None
    raw = _extract_first_json_object(text)

    def _clamp(value) -> Optional[int]:
        try:
            return int(round(max(1.0, min(4.0, float(value)))))
        except (TypeError, ValueError):
            return None

    if raw is not None:
        try:
            obj = json.loads(raw)
            return _clamp(obj.get("soundness_rating", obj.get("rating")))
        except json.JSONDecodeError:
            pass
    m = re.search(r'"(?:soundness_rating|rating)"\s*:\s*([-+]?\d+(?:\.\d+)?)',
                  raw if raw is not None else text)
    return _clamp(m.group(1)) if m else None


def rating_to_reward(rating: Optional[int]) -> Optional[float]:
    """Map a 1-4 rating to a [0,1] reward: (rating - 1) / 3. None -> None."""
    if rating is None:
        return None
    return (rating - 1) / 3.0


# --------------------------------------------------------------------------- #
# Endpoint discovery + decode params -- same shape as common/scijudge.py so the
# soundness judge routes identically (self-hosted vLLM by default; gpt-/claude-/
# gemini- names auto-route through the gateway).
# --------------------------------------------------------------------------- #
def _normalize_base_url(url: str) -> str:
    if urlsplit(url).path in ("", "/"):
        return url.rstrip("/") + "/v1"
    return url


def resolve_base_url(cli_url: Optional[str] = None, endpoint_file: Optional[str] = None) -> str:
    """Resolve the judge's OpenAI-compatible base URL.

    Precedence: explicit ``cli_url`` -> ``$SOUNDNESS_BASE_URL`` -> the ``URL=``
    line in ``endpoint_file`` (default ``$SOUNDNESS_ENDPOINT_FILE``). A bare host
    is normalized to ``/v1``; a URL with an explicit path is kept as-is.
    """
    for url in (cli_url, os.environ.get("SOUNDNESS_BASE_URL", "").strip()):
        if url:
            return _normalize_base_url(url)
    endpoint_file = endpoint_file or os.environ.get("SOUNDNESS_ENDPOINT_FILE", "")
    if endpoint_file and Path(endpoint_file).is_file():
        for line in Path(endpoint_file).read_text().splitlines():
            if line.startswith("URL="):
                return _normalize_base_url(line.split("=", 1)[1].strip())
    raise RuntimeError(
        "No soundness judge endpoint. Set SOUNDNESS_BASE_URL (or pass --base-url) "
        "to a vLLM server or a gpt-/claude- gateway, or write URL= into "
        f"SOUNDNESS_ENDPOINT_FILE (={endpoint_file!r})."
    )


# Models whose names start with "gpt" but are self-hosted on vLLM, so they must
# NEVER route through the SF gateway even though gateway.is_gpt() matches the
# prefix (mirrors common/clients.py::_LOCAL_DESPITE_GPT_PREFIX). gpt-oss-* is the
# default local judge served by serve_judge.sbatch.
_LOCAL_DESPITE_GPT_PREFIX = ("gpt-oss",)


def _looks_local(model: Optional[str]) -> bool:
    name = (model or DEFAULT_MODEL or "").split("/")[-1].lower()
    return any(name.startswith(p) for p in _LOCAL_DESPITE_GPT_PREFIX)


def is_gateway_model(model: Optional[str] = None) -> bool:
    """Whether the judge is a hosted gateway model. Override with
    ``SOUNDNESS_BACKEND=vllm|gateway``. gpt-oss-* is always local despite its
    'gpt' prefix."""
    backend = os.environ.get("SOUNDNESS_BACKEND", "").strip().lower()
    if backend in ("gateway", "openai", "api"):
        return True
    if backend in ("vllm", "local"):
        return False
    if _looks_local(model):
        return False
    return gateway.is_hosted(model or DEFAULT_MODEL)


def judge_client_config(cli_url: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Client kwargs ({base_url, api_key, default_headers}) for the judge.

    Hosted gpt-/claude-/gemini- judges route through the SF/eng-ai gateways with
    KeyHandler keys; only an explicit ``cli_url`` overrides the gateway. Otherwise
    it's a self-hosted vLLM server (key ``"EMPTY"``)."""
    model = model or DEFAULT_MODEL
    # gpt-oss-* is self-hosted despite the 'gpt' prefix -> force the vLLM path so
    # gateway.client_config can't misroute it to the SF gateway (400 "not enabled").
    if not _looks_local(model):
        cfg = gateway.client_config(model, cli_url)
        if cfg is not None:
            return cfg
    return {"base_url": resolve_base_url(cli_url), "api_key": "EMPTY", "default_headers": {}}


def sampling_kwargs(max_tokens: int, temperature: float, model: Optional[str] = None) -> dict:
    """Judge decode params at ``temperature`` (self-consistency ballots sample at
    a non-zero temperature). Honors the same provider quirks as scijudge:
      - vLLM gets top_p + top_k.
      - gpt-5/o-series reject non-default temperature and want max_completion_tokens.
      - gemini-2.5 is a thinking model -> cap the thinking budget so the JSON fits.
    """
    kw = {"temperature": temperature, "max_tokens": max_tokens}
    if is_gateway_model(model):
        name = (model or DEFAULT_MODEL or "").split("/")[-1].lower()
        if name.startswith(("gpt-5", "o1", "o3", "o4")):
            kw.pop("temperature")
            kw["max_completion_tokens"] = kw.pop("max_tokens")
        # Newer Claude (Opus 4.7/4.8, Sonnet 5) 400 on an explicit `temperature`
        # on the eng-ai gateway -- they run at the model default; drop the knob
        # (mirrors the benchmark client's guard in rigorbench/llm/client.py and
        # project memory claude-opus-48-rejects-temperature). NOTE: with no
        # settable temperature the N-vote self-consistency loses its diversity
        # source, so for these judges set SOUNDNESS_VOTES=1 (see the reward doc).
        if "opus-4-8" in name or "opus-4-7" in name or "claude-sonnet-5" in name:
            kw.pop("temperature", None)
        if name.startswith("gemini-2.5"):
            budget = int(os.environ.get("SOUNDNESS_THINKING_BUDGET", "512"))
            if budget >= 0:
                kw["extra_body"] = {"thinkingConfig": {"thinkingBudget": budget}}
    else:
        kw["top_p"] = float(os.environ.get("SOUNDNESS_TOP_P", "0.8"))
        kw["extra_body"] = {"top_k": int(os.environ.get("SOUNDNESS_TOP_K", "20"))}
    return kw
