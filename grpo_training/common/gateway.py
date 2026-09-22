"""Hosted-model (gpt-*/claude-*/gemini-*) routing + keys.

GPT-family models route to ``OPENAI_BASE_URL`` (unset -> the OpenAI SDK's own
public default). Claude/Gemini route to ``ANTHROPIC_BASE_URL`` (unset -> the
Anthropic SDK's own public default) using ``ANTHROPIC_API_KEY``. Set
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` in your environment (see
``.env.example``); there is no other key source.

Used by the judge (reward + eval) and by inference so any of them can target a
gpt-/claude- model just by name; non-hosted models fall back to a local vLLM URL.
"""
from __future__ import annotations

import os
from typing import Optional

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or None
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL") or None

_GPT_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")


def is_gpt(model: Optional[str]) -> bool:
    return (model or "").split("/")[-1].lower().startswith(_GPT_PREFIXES)


def is_claude(model: Optional[str]) -> bool:
    return "claude" in (model or "").lower()


def is_gemini(model: Optional[str]) -> bool:
    return "gemini" in (model or "").lower()


def is_hosted(model: Optional[str]) -> bool:
    """True for models served via the gpt-/claude-/gemini- gateways."""
    return is_gpt(model) or is_claude(model) or is_gemini(model)


def client_config(model: Optional[str], base_url: Optional[str] = None) -> Optional[dict]:
    """Client kwargs ({base_url, api_key, default_headers}) for a hosted gpt-/claude-
    model, or None if ``model`` is not a hosted model (caller uses a local vLLM URL).

    ``base_url`` overrides the default for that family if provided.
    """
    if not is_hosted(model):
        return None
    if is_gpt(model):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                f"No OPENAI_API_KEY set in the environment for GPT model {model!r}. "
                f"See .env.example."
            )
        cfg = {"api_key": key, "default_headers": {}}
        url = base_url or OPENAI_BASE_URL
        if url:
            cfg["base_url"] = url
        return cfg
    family = "Gemini" if is_gemini(model) else "Claude"
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            f"No ANTHROPIC_API_KEY set in the environment for {family} model {model!r}. "
            f"See .env.example."
        )
    cfg = {"api_key": key, "default_headers": {}}
    url = base_url or ANTHROPIC_BASE_URL
    if url:
        cfg["base_url"] = url
    return cfg
