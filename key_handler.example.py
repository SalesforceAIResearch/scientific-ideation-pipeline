"""Optional key-loading shim, kept for parity with internal call sites that used
this import shape historically. Nothing in this release actually depends on it
any more -- gateway.py reads OPENAI_API_KEY / ANTHROPIC_API_KEY straight from
os.environ -- but if you're wiring in a script that still does
``from key_handler import KeyHandler; KeyHandler.set_env_key()``, copy this to
key_handler.py at the repo root and it will keep working, reading from your
environment (or a local .env you've sourced) instead of hardcoded values.

Never commit a key_handler.py with real secrets in it.
"""
from __future__ import annotations

import os


class KeyHandler:
    REQUIRED = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

    @classmethod
    def set_env_key(cls) -> None:
        missing = [name for name in cls.REQUIRED if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                f"Set them directly, or copy .env.example to .env and source it."
            )
