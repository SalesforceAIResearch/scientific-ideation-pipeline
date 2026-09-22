"""Shared SciJudge plumbing (single source of truth for the judge interface).

SciJudge (OpenMOSS-Team/SciJudge-30B) is a pairwise citation-impact judge: given
two papers' title+abstract it picks the higher-cited one via ``<answer>A/B</answer>``.
Both the GRPO reward (rewards/scijudge_vs_gt.py, synchronous) and the head-to-head
eval (eval/scijudge_compare.py, async) talk to it, so the prompt, parser,
endpoint discovery, sampling, and A/B-swap scoring live here once.

The prompt MUST stay verbatim in SciJudge's training distribution (mirrors
../reward/scijudge_eval.py) — keep it here and nowhere else.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from common import gateway

DEFAULT_MODEL = os.environ.get("SCIJUDGE_MODEL", "OpenMOSS-Team/SciJudge-30B")
# Title attached to each insight when we frame it as a "paper" for the judge.
DEFAULT_IDEA_TITLE = os.environ.get("SCIJUDGE_IDEA_TITLE", "A proposed research idea")

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Today is {today}. Below are two research ideas, A and B. Determine which "
    "idea is likely to have higher scientific impact (i.e. would attract more "
    "follow-up work and citations).\n"
    "Show your reasoning process in <reason> </reason> tags. And return the "
    "final answer in <answer> </answer> tags. The final answer should contain "
    "only 'A' or 'B'.\n\n"
    "Idea A:\n"
    "{abstract_a}\n\n"
    "Idea B:\n"
    "{abstract_b}\n"
)

_ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)


def _normalize_base_url(url: str) -> str:
    """Append ``/v1`` only when the URL is a bare host (no path), so a gateway link
    that already carries its own path (e.g. ``https://gw/llm/openai``) is left alone."""
    if urlsplit(url).path in ("", "/"):
        return url.rstrip("/") + "/v1"
    return url


def resolve_base_url(cli_url: Optional[str] = None, endpoint_file: Optional[str] = None) -> str:
    """Resolve the judge's OpenAI-compatible base URL.

    Works for a self-hosted vLLM server OR a hosted gateway (gpt-*/claude-*). A
    bare host is normalized to ``/v1``; a URL with an explicit path is kept as-is.

    Precedence: explicit ``cli_url`` -> ``$SCIJUDGE_BASE_URL`` -> the ``URL=`` line
    in ``endpoint_file`` (default ``$SCIJUDGE_ENDPOINT_FILE``, written by
    serve_scijudge.sbatch).
    """
    candidates = [cli_url, os.environ.get("SCIJUDGE_BASE_URL", "").strip()]
    for url in candidates:
        if url:
            return _normalize_base_url(url)
    endpoint_file = endpoint_file or os.environ.get("SCIJUDGE_ENDPOINT_FILE", "")
    if endpoint_file and Path(endpoint_file).is_file():
        for line in Path(endpoint_file).read_text().splitlines():
            if line.startswith("URL="):
                return _normalize_base_url(line.split("=", 1)[1].strip())
    raise RuntimeError(
        "No judge endpoint. Set SCIJUDGE_BASE_URL (or pass --base-url) to a vLLM "
        "server or a gpt-/claude- gateway, or start serve_scijudge.sbatch so it "
        f"writes URL= into SCIJUDGE_ENDPOINT_FILE (={endpoint_file!r})."
    )


def is_gateway_model(model: Optional[str] = None) -> bool:
    """Whether the judge is a hosted gateway model (no vLLM-only sampling knobs).

    Override the heuristic with ``SCIJUDGE_BACKEND=vllm|gateway``."""
    backend = os.environ.get("SCIJUDGE_BACKEND", "").strip().lower()
    if backend in ("gateway", "openai", "api"):
        return True
    if backend in ("vllm", "local"):
        return False
    return gateway.is_hosted(model or DEFAULT_MODEL)


def judge_client_config(cli_url: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Client kwargs ({base_url, api_key, default_headers}) for the judge.

    Hosted gpt-/claude- judges route through the SF/eng-ai gateways with keys from
    KeyHandler (no judge-specific key); only an explicit ``cli_url`` overrides the
    gateway (so a stale ``$SCIJUDGE_BASE_URL`` can't misroute a gpt-/claude- judge).
    Otherwise it's a self-hosted vLLM server (key ``"EMPTY"``) discovered via
    ``cli_url`` / ``$SCIJUDGE_BASE_URL`` / the endpoint file.
    """
    model = model or DEFAULT_MODEL
    cfg = gateway.client_config(model, cli_url)
    if cfg is not None:
        return cfg
    return {"base_url": resolve_base_url(cli_url), "api_key": "EMPTY", "default_headers": {}}


def build_messages(abstract_a: str, abstract_b: str, today: str,
                   title_a: str = DEFAULT_IDEA_TITLE,
                   title_b: str = DEFAULT_IDEA_TITLE) -> list[dict]:
    """Chat messages asking SciJudge which of two research ideas is higher-impact.

    ``abstract_a``/``abstract_b`` are the full idea strings (motivation + problem
    + insight). ``title_a``/``title_b`` are accepted for backward compatibility
    but no longer rendered (the prompt frames ideas directly, not papers)."""
    prompt = USER_TEMPLATE.format(
        today=today, abstract_a=abstract_a, abstract_b=abstract_b,
    )
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}]


def sampling_kwargs(max_tokens: int = 2048, model: Optional[str] = None) -> dict:
    """Judge decode params (greedy by default), honoring SCIJUDGE_* overrides.

    vLLM accepts the full set (temperature + top_p + top_k). Hosted gateway models
    are pickier, so we send them only what they allow:
      - ``top_k`` is vLLM-only (sent via ``extra_body``) -> dropped for gateway.
      - ``top_p`` + ``temperature`` together are rejected by claude-opus-4-8
        ("cannot both be specified") -> drop ``top_p`` for gateway models.
      - gpt-5/o-series reject any non-default ``temperature`` -> drop it too.
    Sending both to a gateway model surfaces as a 400 BadRequestError."""
    temperature = float(os.environ.get("SCIJUDGE_TEMPERATURE", "0.0"))
    kw = {"temperature": temperature, "max_tokens": max_tokens}
    if is_gateway_model(model):
        # gpt-5 / o1 / o3 / o4 reject temperature != default; keep only max_tokens.
        name = (model or DEFAULT_MODEL or "").split("/")[-1].lower()
        if name.startswith(("gpt-5", "o1", "o3", "o4")):
            kw.pop("temperature")
            # gpt-5.x reasoning models reject `max_tokens` outright — the budget
            # must be passed as `max_completion_tokens` (400 otherwise).
            kw["max_completion_tokens"] = kw.pop("max_tokens")
        # claude-opus-4-8 / claude-sonnet-5 / claude-opus-5 DEPRECATED
        # `temperature` and 400 if it is sent (see project memory
        # `claude-opus-48-rejects-temperature`); drop it. Older opus/sonnet
        # (incl. claude-sonnet-4-6) still accept it. Matters because impact
        # defaults to claude-opus-4-8 on the gateway backend, and because we
        # route sonnet-5 through here as an impact judge.
        if ("claude-opus-4-8" in name
                or "claude-sonnet-5" in name
                or "claude-opus-5" in name):
            kw.pop("temperature", None)
        # gemini-2.5 is a *thinking* model: with an unbounded budget it spends the
        # whole max_tokens reasoning and gets truncated (finish_reason='length')
        # before emitting <answer>, so parse_answer() -> None and the impact silently
        # scores 0.0. Cap the thinking budget so the verdict fits. Set
        # SCIJUDGE_THINKING_BUDGET=-1 to disable the cap (unbounded), 0 to turn
        # thinking off entirely.
        if name.startswith("gemini-2.5"):
            budget = int(os.environ.get("SCIJUDGE_THINKING_BUDGET", "512"))
            if budget >= 0:
                kw["extra_body"] = {"thinkingConfig": {"thinkingBudget": budget}}
    else:
        kw["top_p"] = float(os.environ.get("SCIJUDGE_TOP_P", "0.8"))
        kw["extra_body"] = {"top_k": int(os.environ.get("SCIJUDGE_TOP_K", "20"))}
    return kw


def parse_answer(text: str) -> Optional[str]:
    """Return 'A' / 'B' from a SciJudge response, or None if unparseable."""
    if not text:
        return None
    answers = _ANSWER_RE.findall(text)
    if answers:
        return answers[-1].upper()
    m = re.search(r"answer[:\s]*\**\s*([AB])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _a_win_count(ans_ab: Optional[str], ans_ba: Optional[str]) -> int:
    """How many of the two swapped orderings A won. ``ans_ab`` is the verdict with
    A in slot A; ``ans_ba`` is the verdict with A moved to slot B (so A wins == 'B')."""
    return (1 if ans_ab == "A" else 0) + (1 if ans_ba == "B" else 0)


def winrate_a(ans_ab: Optional[str], ans_ba: Optional[str]) -> float:
    """A's win rate across both orderings -> {0.0, 0.5, 1.0}."""
    return _a_win_count(ans_ab, ans_ba) / 2.0


def decide_winner(ans_ab: Optional[str], ans_ba: Optional[str]) -> str:
    """Position-bias-controlled verdict: 'A' / 'B' only if consistent across the
    swap, else 'tie' (also covers inconsistent or unparseable)."""
    wins = _a_win_count(ans_ab, ans_ba)
    if wins == 2:
        return "A"
    if wins == 0 and ans_ab is not None and ans_ba is not None:
        return "B"
    return "tie"
