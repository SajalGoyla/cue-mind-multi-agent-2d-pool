"""Build policies from short names used in CLIs and experiment configs (v2).

Names: stat (simulation-based reference player), rule (hint-based reference player, v3-A),
b1 (easiest pot; pilot only),
sa (single LLM agent), ma (multi-agent team).

LLM policies accept keyword arguments from experiment configs:
    llm: "ollama" (default, or CUEMIND_LLM) or "fake" (offline tests)
    model: override CUEMIND_OLLAMA_MODEL (default qwen2.5:3b-instruct)
    temperature: sampling temperature (default 0.7)
    offline: true -> cache-only (no model calls)
"""

from __future__ import annotations

import os
from typing import Any

from cuemind.policies.rule_bot import RuleBot
from cuemind.policies.search_bot import SearchBot
from cuemind.policies.simple_bots import GeometricBot


def make_llm_client(**kw: Any):
    from cuemind.llm.cache import ResponseCache
    from cuemind.llm.client import FakeLLM, OllamaClient

    kind = kw.get("llm") or os.environ.get("CUEMIND_LLM", "ollama")
    if kind == "fake":
        return FakeLLM(kw.get("fake_mode", "best"))
    if kind != "ollama":
        raise ValueError(f"unknown llm kind {kind!r}; use ollama or fake")
    cache = ResponseCache(kw.get("cache_path") or os.environ.get(
        "CUEMIND_LLM_CACHE", "results/cache/llm_cache.sqlite"))
    return OllamaClient(kw.get("model"), temperature=float(kw.get("temperature", 0.7)),
                        cache=cache, offline=bool(kw.get("offline", False)))


def _sa(**kw):
    from cuemind.policies.single_agent import SingleAgent

    return SingleAgent(make_llm_client(**kw))


def _ma(**kw):
    from cuemind.policies.multi_agent import MultiAgent

    return MultiAgent(make_llm_client(**kw))


_BUILDERS = {
    "stat": lambda **kw: SearchBot(),
    "b1": lambda **kw: GeometricBot(),
    "rule": lambda **kw: RuleBot(),
    "sa": _sa,
    "ma": _ma,
}


def make_policy(name: str, **kwargs: Any):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[key](**kwargs)


def known_policies() -> list[str]:
    return sorted(_BUILDERS)
