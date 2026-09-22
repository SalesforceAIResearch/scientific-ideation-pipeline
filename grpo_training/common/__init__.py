"""Shared helpers used across rewards/, eval/, and inference/.

Kept transport-agnostic (no OpenAI client lives here) so a synchronous reward
and an async eval can both reuse the same prompt, parser, endpoint discovery, and
win-rate logic without dragging in each other's execution model.
"""
