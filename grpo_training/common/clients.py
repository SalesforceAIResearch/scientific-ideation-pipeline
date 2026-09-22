"""Async OpenAI-compatible client factory with a `--backend vllm|gateway` switch.

Single place the evaluators build their judge client, so the novelty / grounding
/ impact judges share identical routing, key handling, TLS, and per-model
generation-config quirks. Transport lives here (this module DOES import the
OpenAI client); the rubric/prompt/parse logic stays pure in common/{novelty,
grounding,inspire,scijudge}.py.

Two backends:
  - "gateway": hosted gpt-*/claude-*/gemini- models, routed via common/gateway.py
    using OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment (see
    .env.example). If your provider sits behind a custom CA, point
    CUSTOM_CA_BUNDLE at it and it will be applied before the client is built.
  - "vllm": a self-hosted vLLM OpenAI server (gpt-oss-120b / SciJudge-30B),
    discovered via an explicit URL or the *_ENDPOINT_FILE written by the serve
    sbatch jobs. api_key is the dummy "EMPTY" vLLM ignores.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

# Repo root + grpo_training importable (mirrors common/scijudge.py).
_GRPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GRPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_GRPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import gateway          # noqa: E402  (hosted-model routing + keys)
from common import scijudge         # noqa: E402  (resolve_base_url for vllm)

# Optional custom CA bundle for gateway TLS from Python (curl may work without
# it via the system trust store; Python's ssl module needs it explicit). Only
# applied if CUSTOM_CA_BUNDLE is set and points at a real file -- no default.
_CA_BUNDLE_CANDIDATES = [
    os.environ.get("CUSTOM_CA_BUNDLE", ""),
]

VALID_BACKENDS = ("vllm", "gateway", "bedrock")

# Models whose names start with "gpt" but are actually self-hosted on vLLM, so
# they must NEVER route through the hosted gateway even though gateway.is_gpt()
# matches their prefix. gpt-oss-* is the local judge served by serve_judge.sbatch.
_LOCAL_DESPITE_GPT_PREFIX = ("gpt-oss",)

# Model-name markers for Chinese-lab models (DeepSeek, Qwen, Kimi/Moonshot,
# GLM/Zhipu, MiniMax, Yi, Baichuan, ...) -- vendor-family flag alongside
# gateway.is_gpt/is_claude/is_gemini. These share two routing quirks worth
# flagging together: (1) on Bedrock they are converse-only, no /openai/v1 route
# (mirrors bedrock_converse.py's is_anthropic_model split); (2) the ones that
# reason write the chain of thought INLINE in `content` rather than a separate
# reasoning_content field, so it is billed against the same max_tokens budget as
# the answer -- see generation_config()'s floor below and the "fact 3" note in
# bedrock_converse.py (a tight cap truncates the JSON before the rating is ever
# written, same failure mode Sonnet-5-on-Bedrock has).
_CHINESE_MODEL_MARKERS = ("deepseek", "qwen", "kimi", "moonshot", "glm", "zhipu",
                          "zai", "minimax", "yi-", "baichuan")


def _looks_local(model: str) -> bool:
    name = (model or "").split("/")[-1].lower()
    return any(name.startswith(p) for p in _LOCAL_DESPITE_GPT_PREFIX)


def is_chinese_model(model: Optional[str]) -> bool:
    """True for a model from a Chinese lab, by name marker (substring, same
    simplistic check as gateway.is_claude/is_gemini) -- matches bare ids
    ("deepseek-r1"), Bedrock-provider-prefixed ids ("deepseek.r1-v1:0"), and
    inference-profile-prefixed ids ("us.deepseek.r1-v1:0")."""
    name = (model or "").lower()
    return any(marker in name for marker in _CHINESE_MODEL_MARKERS)


def _apply_ca_bundle() -> Optional[str]:
    """Point Python's TLS stack at CUSTOM_CA_BUNDLE if it's set and exists.
    Returns the path used, or None (gateway TLS may then fail -- caller's risk)."""
    for cand in _CA_BUNDLE_CANDIDATES:
        if cand and Path(cand).is_file():
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cand)
            os.environ.setdefault("SSL_CERT_FILE", cand)
            return cand
    return None


def generation_config(model: str, max_tokens: int) -> dict:
    """Per-model decode kwargs, normalizing the annoying provider differences.

    - claude-*: only max_tokens (no top_p needed).
    - gemini-*: reasoning models that spend hidden "thinking" tokens out of the
      SAME completion budget, so a tight max_tokens truncates the visible answer
      mid-string (finish_reason=length). Give them a generous floor so the JSON
      body always fits after the thinking.
    - gpt-5 / gpt-5.x: reasoning models -- use max_completion_tokens, and they
      REJECT temperature != 1 and top_p (caller must also skip temperature).
    - Chinese-lab models (DeepSeek/Qwen/Kimi/GLM/...): same failure mode as
      gemini -- reasoning is written INLINE in the visible content (no separate
      reasoning_content field on Bedrock converse), billed against the same
      max_tokens budget, so give them the same generous floor.
    - everything else (gpt-4*, vLLM-served models): max_tokens + top_p.
    """
    name = (model or "").lower()
    if "claude" in name:
        return {"max_tokens": max_tokens}
    if "gemini" in name:
        return {"max_tokens": max(max_tokens, 4096)}
    if "gpt-5" in name:
        return {"max_completion_tokens": max_tokens}
    if is_chinese_model(model):
        return {"max_tokens": max(max_tokens, 4096), "top_p": 1.0}
    return {"max_tokens": max_tokens, "top_p": 1.0}


def accepts_temperature(model: str) -> bool:
    """Whether a model accepts an explicit `temperature`.

    - gpt-5 / gpt-5.x reject any temperature other than the default.
    - The newer Claude tier -- opus-4-8, opus-4-9, and sonnet-5 -- DEPRECATED
      `temperature` and 400s if it is sent (see project memory
      `claude-opus-48-rejects-temperature`, and summarize_document.py's family
      table); older opus/sonnet (e.g. sonnet-4-6) still take it. This matters
      because every judge here (novelty/diversity/feasibility) defaults to a
      newer-tier model.
    """
    name = (model or "").lower()
    _no_temp = ("gpt-5", "opus-4-8", "opus-4-9", "sonnet-5")
    return not any(tok in name for tok in _no_temp)


class _BedrockUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _BedrockMessage:
    def __init__(self, content: Optional[str]):
        self.content = content


class _BedrockChoice:
    def __init__(self, content: Optional[str]):
        self.message = _BedrockMessage(content)


class _BedrockChatCompletion:
    """Fake ``ChatCompletion``-shaped response over a converse result, exposing
    only what the judge call sites read: ``.choices[0].message.content`` and
    ``.usage.{prompt,completion,total}_tokens`` (the latter via usage_dict())."""

    def __init__(self, result: Dict[str, Any]):
        self.choices = [_BedrockChoice(result.get("text"))]
        self.usage = _BedrockUsage(result.get("prompt_tokens"),
                                    result.get("completion_tokens"),
                                    result.get("total_tokens"))
        self.stop_reason = result.get("stop_reason")


class _BedrockChatCompletions:
    def __init__(self, region: Optional[str]):
        self._region = region

    async def create(self, *, model: str, messages, max_tokens: Optional[int] = None,
                     temperature: Optional[float] = None, **_ignored
                     ) -> _BedrockChatCompletion:
        # **_ignored swallows kwargs converse has no equivalent for (e.g. top_p,
        # which generation_config() sets for non-Claude models but Bedrock's
        # inferenceConfig does not accept here) rather than 400ing on them.
        from common import bedrock_converse
        kw: Dict[str, Any] = {"model": model, "region": self._region}
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        if temperature is not None:
            kw["temperature"] = temperature
        result = await bedrock_converse.converse_async_full(messages, **kw)
        # RAISE on a failed call instead of returning a null-text response.
        # bedrock_converse deliberately never raises (its reward call sites impute a
        # missing score), but this adapter impersonates AsyncOpenAI, whose callers
        # expect an exception on an API error and wrap the call in try/except. Passing
        # text=None through instead makes a transport failure indistinguishable from
        # a genuinely unparseable judgement: the judge parses "" -> rating None, the
        # harness logs error=None, and the run reports a confident all-null result.
        # (Measured 2026-08-13: a mangled model id -- a caller's shell had stripped
        # `qwen.qwen3-235b-a22b-2507-v1:0` down to the literal "0" -- produced a
        # ValidationException on all 30 instances that surfaced as a clean 30/30-null
        # agreement table rather than as an error. Any converse-layer failure does
        # this: throttling, a bad region, expired credentials.)
        if result.get("text") is None:
            raise RuntimeError(
                f"bedrock converse failed for {model}: {result.get('stop_reason')}")
        return _BedrockChatCompletion(result)


class BedrockConverseClient:
    """Minimal ``AsyncOpenAI``-shaped adapter over Bedrock's native converse API,
    for models with no ``/openai/v1`` route -- Claude and Chinese-lab reasoning
    models such as DeepSeek-R1 (see bedrock_converse.py's module docstring for why
    converse, not the OpenAI-compatible route, is required for these). Only
    implements ``.chat.completions.create(...)``, the one surface the judges call.
    """

    def __init__(self, region: Optional[str] = None):
        self.chat = SimpleNamespace(completions=_BedrockChatCompletions(region))


def build_async_client(
    backend: str,
    model: str,
    *,
    base_url: Optional[str] = None,
    endpoint_file: Optional[str] = None,
    timeout: Optional[float] = None,
):
    """Construct a judge client for the chosen backend.

    gateway: route gpt-/claude-/gemini- to their public (or `base_url`-overridden)
             endpoint using OPENAI_API_KEY / ANTHROPIC_API_KEY from the
             environment. `base_url` overrides the default for that family.
             Returns AsyncOpenAI.
    vllm:    resolve a self-hosted vLLM URL from `base_url` -> $SCIJUDGE_BASE_URL
             -> the endpoint file; api_key="EMPTY". Returns AsyncOpenAI.
    bedrock: AWS Bedrock's native converse API (SigV4 via boto3, not an HTTP
             base_url), for models with no OpenAI-compatible route -- e.g.
             Claude on Bedrock (`us.anthropic.claude-sonnet-5`) or DeepSeek-R1
             (`us.deepseek.r1-v1:0`). `base_url`/`endpoint_file` are ignored;
             region comes from $BEDROCK_REGION / $AWS_REGION (see
             bedrock_converse.DEFAULT_REGION). Returns BedrockConverseClient,
             which exposes the same `.chat.completions.create(...)` surface the
             judges already call, so no call site needs a bedrock-specific branch.
    """
    backend = (backend or "").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {backend!r}")

    kwargs = {} if timeout is None else {"timeout": timeout}

    if backend == "gateway":
        if _looks_local(model):
            raise ValueError(
                f"backend='gateway' but model {model!r} is a self-hosted vLLM judge "
                f"(its 'gpt' prefix is misleading). Use --backend vllm for it."
            )
        # If you've set CUSTOM_CA_BUNDLE (e.g. your org's gateway sits behind a
        # private CA), apply it before building the client. Left unset, this is
        # a no-op and the SDK's default TLS trust store is used.
        _apply_ca_bundle()
        cfg = gateway.client_config(model, base_url)
        if cfg is None:
            raise ValueError(
                f"backend='gateway' but model {model!r} is not a hosted gpt-/claude- "
                f"model. Use --backend vllm for self-hosted models, or pass a hosted "
                f"model name."
            )
        # Gateway enforces ~100 req/min; high --max-concurrent bursts trip a 429.
        # The SDK retries 429s with exponential backoff, but its default cap of 2 is
        # too low for a burst -- raise it so transient rate-limits self-heal instead
        # of surfacing as a null/lost judge score. Override via SCIJUDGE_MAX_RETRIES.
        max_retries = int(os.environ.get("SCIJUDGE_MAX_RETRIES", "8"))
        return AsyncOpenAI(**cfg, max_retries=max_retries, **kwargs)

    if backend == "bedrock":
        region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")
        return BedrockConverseClient(region=region)

    # backend == "vllm"
    url = scijudge.resolve_base_url(base_url, endpoint_file)
    return AsyncOpenAI(base_url=url, api_key="EMPTY", default_headers={}, **kwargs)


# --------------------------------------------------------------------------- #
# Usage / cost accounting (shared by all judges so per-feature metadata is
# computed identically). Local vLLM has no price -> cost stays None unless the
# caller passes $/1M-token rates.
# --------------------------------------------------------------------------- #
def usage_dict(resp) -> Dict[str, Optional[int]]:
    """Pull prompt/completion/total token counts off an OpenAI-compatible
    response (vLLM and the gateways both populate ``resp.usage``). Returns Nones
    if usage is absent."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
    }


def token_cost(usage: Dict[str, Optional[int]], *, price_in: Optional[float],
               price_out: Optional[float]) -> Optional[float]:
    """Dollar cost from a usage dict given $/1M-token rates. Returns None when no
    rates are supplied (e.g. local vLLM) so cost is explicitly 'not priced'
    rather than a misleading 0.0."""
    if price_in is None and price_out is None:
        return None
    pin = price_in or 0.0
    pout = price_out or 0.0
    p_tok = usage.get("prompt_tokens") or 0
    c_tok = usage.get("completion_tokens") or 0
    return (p_tok * pin + c_tok * pout) / 1_000_000.0


def add_usage(*usages: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    """Sum several usage dicts (Nones treated as 0); total is None only if every
    contributing field was None."""
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = {k: False for k in out}
    for u in usages:
        for k in out:
            v = u.get(k)
            if v is not None:
                out[k] += v
                seen[k] = True
    return {k: (out[k] if seen[k] else None) for k in out}
