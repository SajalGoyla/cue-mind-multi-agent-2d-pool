"""LLM layer and agents, tested offline with FakeLLM (no API calls)."""

import pytest

from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.llm import promptlib as P
from cuemind.llm.cache import Budget, BudgetExceeded, ResponseCache, request_key
from cuemind.llm.client import ClaudeVertexClient, FakeLLM, cost_usd
from cuemind.llm.schemas import SingleDecision
from cuemind.policies.base import DecisionContext
from cuemind.policies.multi_agent import MultiAgent
from cuemind.policies.registry import make_policy
from cuemind.policies.single_agent import SingleAgent
from cuemind.sim.state import TableState

MID = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "4": (0.3, 0.9),
                                         "10": (0.5, 1.0), "12": (0.8, 1.7), "8": (0.3, 1.3)},
                 groups={"P1": "solids", "P2": "stripes"})


@pytest.fixture(scope="module")
def ctx():
    cands = prepare(MID, PrepConfig(access_level="L2", n_rollouts=4), rollout_key=(5,))
    return DecisionContext(state=MID, candidates=cands,
                           history=[{"seat": "P2", "shot_id": "S03", "intent": "safety",
                                     "result": "nothing potted"}])


def best_by_p_pot(ctx):
    return max(ctx.candidates, key=lambda c: (c.rollouts.p_pot, -int(c.shot_id[1:]))).shot_id


# ------------------------------------------------------------------ fairness of prompts
def test_single_agent_prompt_contains_all_specialist_knowledge():
    single = P.system_single()
    for p in P.PERSPECTIVES:
        assert P.knowledge(p) in single
        spec = P.system_specialist(p)
        assert P.knowledge(p) in spec
        for other in set(P.PERSPECTIVES) - {p}:
            assert P.knowledge(other) not in spec
    assert "{perspective}" not in P.system_specialist("offense")
    assert len(P.prompt_version()) == 12


def test_context_text_lists_every_candidate(ctx):
    text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history)
    for c in ctx.candidates:
        assert f"{c.shot_id} | {c.description}" in text
    assert "Opponent: safety (S03)" in text and "None" not in text


# ------------------------------------------------------------------ single agent
def test_single_agent_decides_validly(ctx):
    llm = FakeLLM("best")
    d = SingleAgent(llm).decide(ctx)
    assert d.shot_id == best_by_p_pot(ctx) and not d.fallback
    assert d.usage["calls"] == 1 and len(llm.calls) == 1
    assert d.trace["agents"][0]["parsed"]["shot_id"] == d.shot_id


def test_invalid_ids_retry_once_then_fall_back(ctx):
    llm = FakeLLM("invalid")
    d = SingleAgent(llm).decide(ctx)
    assert d.fallback and len(llm.calls) == 2 and d.usage["calls"] == 2
    d2 = SingleAgent(FakeLLM("error")).decide(ctx)
    assert d2.fallback


def test_sa_vote_uses_four_samples(ctx):
    llm = FakeLLM("best")
    d = SingleAgent(llm, n_votes=4).decide(ctx)
    assert [c["repeat_idx"] for c in llm.calls] == [0, 1, 2, 3]
    assert d.trace["votes"] == {d.shot_id: 4}


# ------------------------------------------------------------------ multi agent
def test_multi_agent_graph_fans_out_and_coordinates(ctx):
    llm = FakeLLM("best")
    ma = MultiAgent(llm)
    d = ma.decide(ctx)
    agents = sorted(c["agent"] for c in llm.calls)
    assert agents == ["coordinator", "offense", "position", "safety"]
    assert set(d.trace["reports"]) == {"offense", "position", "safety"}
    assert d.shot_id == best_by_p_pot(ctx) and not d.fallback and d.usage["calls"] == 4
    assert set(d.trace["followed"]) == {"offense", "position", "safety"}


def test_multi_agent_ablation_and_critic(ctx):
    llm = FakeLLM("best")
    ma = MultiAgent(llm, specialists=("offense", "safety"))
    assert ma.name == "MA-no-position"
    ma.decide(ctx)
    assert sorted(c["agent"] for c in llm.calls) == ["coordinator", "offense", "safety"]
    llm2 = FakeLLM("best")
    d = MultiAgent(llm2, critic=True).decide(ctx)
    assert "critic" in [c["agent"] for c in llm2.calls] and d.trace["critic"]["approve"]


def test_multi_agent_falls_back_when_coordinator_fails(ctx):
    d = MultiAgent(FakeLLM("invalid")).decide(ctx)
    assert d.fallback


def test_registry_builds_llm_policies_with_fake_llm():
    for name in ("sa", "sa-vote", "ma", "ma+critic", "ma-no-safety"):
        p = make_policy(name, llm="fake")
        assert hasattr(p, "decide")


# ------------------------------------------------------------------ cache, budget, cost
def test_cache_hit_and_offline_mode(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    client = ClaudeVertexClient("claude-haiku-5-5", "medium", cache=cache, offline=True,
                                project_id="x")
    miss = client.complete(system="s", user="u", schema=SingleDecision, agent="single")
    assert miss.parsed is None and "offline" in miss.error
    key = request_key(model="claude-haiku-5-5", effort="medium", system="s", user="u",
                      schema="SingleDecision", repeat_idx=0)
    cache.put(key, {"parsed": {"shot_id": "S01", "intent": "offense", "rationale": "r",
                               "confidence": 0.5}, "usage": {"cost_usd": 0.001}})
    hit = client.complete(system="s", user="u", schema=SingleDecision, agent="single")
    assert hit.cached and hit.parsed.shot_id == "S01" and len(cache) == 1


def test_budget_guard(tmp_path):
    b = Budget(tmp_path / "b.sqlite", "run1", max_usd=0.01)
    b.check()
    b.add(0.006)
    b.add(0.005)
    with pytest.raises(BudgetExceeded):
        b.check()
    assert Budget(tmp_path / "b.sqlite", "run2", 0.01).spent() == 0.0


def test_cost_formula():
    u = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_tokens": 0}
    assert cost_usd("claude-haiku-5-5", u) == pytest.approx(0.60)
    assert cost_usd("unknown-model", u) == 0.0


# ------------------------------------------------------------------ ollama (mocked HTTP)
def test_ollama_client_parses_caches_and_reports_errors(tmp_path, monkeypatch):
    import json as _json

    import httpx

    from cuemind.llm.client import OllamaClient

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
    assert r.parsed.shot_id == "S02" and r.usage["input_tokens"] == 900 and r.usage["cost_usd"] == 0
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

    # a server that fails twice and then answers is retried transparently
    calls = {"n": 0}

    def flaky(url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("restarting")
        return Resp()

    monkeypatch.setattr(httpx, "post", flaky)
    ok = OllamaClient("qwen-test").complete(system="s", user="y", schema=SingleDecision, agent="a")
    assert ok.parsed is not None and calls["n"] == 3


def test_confidence_percentages_are_normalised():
    from cuemind.llm.schemas import clamp01

    assert clamp01(0.4) == 0.4 and clamp01(95) == 0.95 and clamp01(-3) == 0.0
    assert clamp01(500) == 1.0 and clamp01("x") == 0.5
