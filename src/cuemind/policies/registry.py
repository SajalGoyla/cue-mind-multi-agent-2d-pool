"""Build policies from short names used in CLIs and experiment configs.

Names: b0, b1, b2 (bots); sa, sa-vote (4 votes); ma, ma+critic, ma-no-offense,
ma-no-position, ma-no-safety (LLM systems).

LLM policies accept keyword arguments (from experiment configs):
    llm: "claude" (default) or "fake"
    model, effort: override CUEMIND_MODEL / CUEMIND_EFFORT
    run_id, max_usd: budget guard shared by all worker processes
    offline: true -> cache-only (no API calls)
"""

from __future__ import annotations

import os
from typing import Any

from cuemind.policies.search_bot import SearchBot
from cuemind.policies.simple_bots import GeometricBot, RandomBot


def make_llm_client(**kw: Any):
    from cuemind.llm.cache import Budget, ResponseCache
    from cuemind.llm.client import ClaudeVertexClient, FakeLLM

    if kw.get("llm", "claude") == "fake":
        return FakeLLM(kw.get("fake_mode", "best"))
    cache_path = kw.get("cache_path") or os.environ.get("CUEMIND_LLM_CACHE",
                                                        "results/cache/llm_cache.sqlite")
    cache = ResponseCache(cache_path)
    budget = None
    if kw.get("max_usd") is not None:
        budget = Budget(cache_path, kw.get("run_id", "default"), float(kw["max_usd"]))
    return ClaudeVertexClient(kw.get("model"), kw.get("effort"), cache=cache, budget=budget,
                              offline=bool(kw.get("offline", False)))


def _sa(n_votes: int):
    def build(**kw):
        from cuemind.policies.single_agent import SingleAgent

        return SingleAgent(make_llm_client(**kw), n_votes=n_votes)
    return build


def _ma(specialists: tuple[str, ...] = ("offense", "position", "safety"), critic: bool = False):
    def build(**kw):
        from cuemind.policies.multi_agent import MultiAgent

        return MultiAgent(make_llm_client(**kw), specialists=specialists, critic=critic)
    return build


_BUILDERS = {
    "b0": lambda **kw: RandomBot(),
    "b1": lambda **kw: GeometricBot(),
    "b2": lambda **kw: SearchBot(kw.get("weights")),
    "sa": _sa(1),
    "sa-vote": _sa(4),
    "ma": _ma(),
    "ma+critic": _ma(critic=True),
    "ma-no-offense": _ma(("position", "safety")),
    "ma-no-position": _ma(("offense", "safety")),
    "ma-no-safety": _ma(("offense", "position")),
}


def register(name: str, builder) -> None:
    _BUILDERS[name] = builder


def make_policy(name: str, **kwargs: Any):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[key](**kwargs)


def known_policies() -> list[str]:
    return sorted(_BUILDERS)
