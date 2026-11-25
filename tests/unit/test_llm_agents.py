"""LLM layer and agents, tested offline with FakeLLM (no model server needed)."""

import pytest

from cuemind.candidates.describe import ease_word
from cuemind.candidates.pipeline import build_menu
from cuemind.llm import promptlib as P
from cuemind.llm.cache import ResponseCache, request_key
from cuemind.llm.client import FakeLLM, OllamaClient
from cuemind.llm.schemas import SingleDecision
from cuemind.policies.base import DecisionContext
from cuemind.policies.multi_agent import MultiAgent
from cuemind.policies.registry import make_policy
from cuemind.policies.single_agent import SingleAgent
from cuemind.sim.state import TableState

MID = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "4": (0.3, 0.9),
                                         "10": (0.5, 1.0), "12": (0.8, 1.7), "8": (0.3, 1.3)},
                 groups={"P1": "solids", "P2": "stripes"})
WORD_RANK = {"easy": 3, "moderate": 2, "hard": 1}


@pytest.fixture(scope="module")
def ctx():
    return DecisionContext(state=MID, candidates=build_menu(MID, menu_key=(5,)),
                           history=[{"seat": "P2", "shot_id": "S03", "intent": "safety",
                                     "result": "nothing potted"}])


def fake_choice(ctx) -> str:
    """The shot FakeLLM('best') picks: best difficulty word, then lowest id."""
    def score(c):
        return WORD_RANK.get(ease_word(c.geometry.ease), 0) if c.shot_type == "pot" else 0
    return min(ctx.candidates, key=lambda c: (-score(c), c.shot_id)).shot_id


# ------------------------------------------------------------------ prompts and fairness
def test_single_agent_prompt_contains_all_specialist_knowledge():
    single = P.system_single()
    for p in P.PERSPECTIVES:
        assert P.knowledge(p) in single
        spec = P.system_specialist(p)
        assert P.knowledge(p) in spec and "{perspective}" not in spec and "{k}" not in spec
        for other in set(P.PERSPECTIVES) - {p}:
            assert P.knowledge(other) not in spec
    assert len(P.prompt_version()) == 12


def test_context_text_lists_every_candidate_and_no_statistics(ctx):
    text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history)
    for c in ctx.candidates:
        assert f"{c.shot_id} | {c.description}" in text
    assert "Opponent: safety (S03)" in text and "None" not in text and "%" not in text


# ------------------------------------------------------------------ single agent
def test_single_agent_decides_validly(ctx):
    llm = FakeLLM("best")
    d = SingleAgent(llm).decide(ctx)
    assert d.shot_id == fake_choice(ctx) and not d.fallback
    assert d.usage["calls"] == 1 and len(llm.calls) == 1


def test_invalid_ids_retry_once_then_fall_back(ctx):
    llm = FakeLLM("invalid")
    d = SingleAgent(llm).decide(ctx)
    assert d.fallback and len(llm.calls) == 2 and d.usage["calls"] == 2
    assert SingleAgent(FakeLLM("error")).decide(ctx).fallback


# ------------------------------------------------------------------ multi agent v2
def test_unanimous_specialists_skip_the_coordinator(ctx):
    llm = FakeLLM("best")
    d = MultiAgent(llm).decide(ctx)
    assert sorted(c["agent"] for c in llm.calls) == ["defense", "offense", "position"]
    assert d.trace["unanimous"] and d.shot_id == fake_choice(ctx) and d.usage["calls"] == 3
    for rep in d.trace["reports"].values():
        assert 1 <= len(rep["shortlist"]) <= P.SHORTLIST_K


def test_disagreement_goes_to_the_coordinator_who_picks_from_the_shortlists(ctx):
    llm = FakeLLM("disagree")
    d = MultiAgent(llm).decide(ctx)
    assert "coordinator" in [c["agent"] for c in llm.calls] and not d.trace["unanimous"]
    assert d.shot_id in d.trace["shortlisted"] and len(d.trace["shortlisted"]) >= 2
    assert not d.fallback and d.usage["calls"] == 4


def test_multi_agent_falls_back_when_nothing_is_valid(ctx):
    d = MultiAgent(FakeLLM("invalid")).decide(ctx)
    assert d.fallback and all("error" in r for r in d.trace["reports"].values())


def test_coordinator_prompt_shows_only_shortlisted_shots(ctx):
    c1, c2 = ctx.candidates[:2]
    text = P.user_coordinator("## Position\n...", [c1], {
        "offense": {"shortlist": [{"shot_id": c1.shot_id, "reason": "easy"}]},
        "defense": {"error": "failed"}})
    assert f"{c1.shot_id} | {c1.description}" in text
    assert f"{c2.shot_id} | " not in text and "(no report: failed)" in text


def test_registry_builds_every_policy():
    for name in ("b", "b1", "sa", "ma"):
        assert hasattr(make_policy(name, llm="fake"), "decide")
    with pytest.raises(ValueError):
        make_policy("ma+critic")


# ------------------------------------------------------------------ cache and ollama client
def test_cache_hit_and_offline_mode(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    client = OllamaClient("m", cache=cache, offline=True)
    miss = client.complete(system="s", user="u", schema=SingleDecision, agent="single")
    assert miss.parsed is None and "offline" in miss.error
    key = request_key(model="m", effort="T0.7", system="s", user="u", schema="SingleDecision",
                      repeat_idx=0)
    cache.put(key, {"parsed": {"shot_id": "S01", "intent": "offense", "rationale": "r",
                               "confidence": 0.5}, "usage": {}})
    hit = client.complete(system="s", user="u", schema=SingleDecision, agent="single")
    assert hit.cached and hit.parsed.shot_id == "S01" and len(cache) == 1


def test_ollama_client_parses_caches_and_reports_errors(tmp_path, monkeypatch):
    import json as _json

    import httpx

    sent = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": _json.dumps({"shot_id": "S02", "intent": "position",
                                                         "rationale": "leave", "confidence": 0.8})},
                    "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient("qwen-test", cache=ResponseCache(tmp_path / "c.sqlite"))
    r = client.complete(system="s", user="u", schema=SingleDecision, agent="single", repeat_idx=2)
    assert r.parsed.shot_id == "S02" and r.usage["input_tokens"] == 900
    assert sent["url"].endswith("/api/chat") and sent["body"]["options"]["seed"] == 1002
    assert sent["body"]["format"]["properties"]["shot_id"]
    again = client.complete(system="s", user="u", schema=SingleDecision, agent="single", repeat_idx=2)
    assert again.cached and again.parsed.shot_id == "S02"

    def refuse(url, json=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(OllamaClient, "RETRY_DELAYS_S", (0, 0))
    bad = OllamaClient("qwen-test").complete(system="s", user="x", schema=SingleDecision, agent="a")
    assert bad.parsed is None and "ollama serve" in bad.error

    calls = {"n": 0}

    def flaky(url, json=None, timeout=None):  # a restarting server is retried transparently
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("restarting")
        if calls["n"] == 2:
            raise httpx.ReadError("connection forcibly closed")
        return Resp()

    monkeypatch.setattr(httpx, "post", flaky)
    ok = OllamaClient("qwen-test").complete(system="s", user="y", schema=SingleDecision, agent="a")
    assert ok.parsed is not None and calls["n"] == 3


def test_confidence_percentages_are_normalised():
    from cuemind.llm.schemas import clamp01

    assert clamp01(0.4) == 0.4 and clamp01(95) == 0.95 and clamp01(-3) == 0.0
    assert clamp01(500) == 1.0 and clamp01("x") == 0.5
