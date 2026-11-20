"""Build policies from short names used in CLIs and experiment configs."""

from __future__ import annotations

from typing import Any

from cuemind.policies.search_bot import SearchBot
from cuemind.policies.simple_bots import GeometricBot, RandomBot

_BUILDERS = {
    "b0": lambda **kw: RandomBot(),
    "b1": lambda **kw: GeometricBot(),
    "b2": lambda **kw: SearchBot(kw.get("weights")),
}


def register(name: str, builder) -> None:
    _BUILDERS[name] = builder


def make_policy(name: str, **kwargs: Any):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[key](**kwargs)
