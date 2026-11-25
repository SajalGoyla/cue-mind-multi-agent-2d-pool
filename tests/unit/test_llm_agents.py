"""LLM layer and agents (v2), tested offline with fake clients (no model calls)."""

import re

import pytest

from cuemind.agents.graph_multi import clean_picks
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.llm import promptlib as P
from cuemind.llm.cache import ResponseCache
from cuemind.llm.client import FakeLLM, LLMResult, OllamaClient
from cuemind.llm.schemas import CoordinatorDecision, Pick, Shortlist, SingleDecision
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
    cands = prepare(MID, PrepConfig(), rollout_key=(5, 1))
    return DecisionContext(state=MID, candidates=cands,
                           history=[{"seat": "P2", "shot_id": "S03", "intent": "safety",
                                     "result": "nothing potted"}])


class Scripted:
    """Fake client: each agent returns a scripted shortlist / decision."""

    model = "scripted"

    def __init__(self, firsts: dict[str, str], coord: str | None = None):
        self.firsts, self.coord, self.calls = firsts, coord, []

    def complete(self, *, system, user, schema, agent, repeat_idx=0):
        self.calls.append(agent)
        ids = re.findall(r"^(S\d{2}) \|", user, re.M)
        if schema is Shortlist:
            first = self.firsts[agent]
            rest = [i for i in ids if i != first][:2]
            obj = Shortlist(picks=[Pick(shot_id=i, reason=f"{agent} reason") for i in [first, *rest]])
        elif schema is CoordinatorDecision:
            obj = CoordinatorDecision(shot_id=self.coord, rationale="coord", followed=["position"])
        else:
            obj = SingleDecision(shot_id=self.firsts.get("single", ids[0]), rationale="r")
        return LLMResult(obj, "", "stop", {"input_tokens": 100, "output_tokens": 10}, 0.0)


# ------------------------------------------------------------------ prompts
def test_single_agent_prompt_contains_all_specialist_knowledge():
    single = P.system_single()
    for p in P.PERSPECTIVES:
        assert P.knowledge(p) in single
        spec = P.system_specialist(p)
        assert P.knowledge(p) in spec
        for other in set(P.PERSPECTIVES) - {p}:
            assert P.knowledge(other) not in spec
    assert P.knowledge("safety") in P.system_coordinator()
    assert "{perspective}" not in P.system_specialist("offense")


def test_llm_text_contains_no_forecast_numbers(ctx):
    text = P.decision_context_text(ctx.state, ctx.candidates, ctx.history)
    for c in ctx.candidates:
        assert f"{c.shot_id} | {c.description}" in text
    assert "%" not in text and "realistic strokes" not in text and "preview" not in text
    assert "Opponent: safety (S03)" in text


# ------------------------------------------------------------------ single agent
def test_single_agent_decides_validly(ctx):
    llm = FakeLLM("best")
    d = SingleAgent(llm).decide(ctx)
    assert ctx.by_id(d.shot_id) is not None and not d.fallback and d.usage["calls"] == 1


def test_invalid_answers_retry_once_then_geometry_fallback(ctx):
    llm = FakeLLM("invalid")
    d = SingleAgent(llm).decide(ctx)
    assert d.fallback and len(llm.calls) == 2
    fb = ctx.by_id(d.shot_id)
    pots = [c for c in ctx.candidates if c.shot_type == "pot"]
    assert fb.shot_type == "pot" and fb.geometry.ease == max(c.geometry.ease for c in pots)
    assert SingleAgent(FakeLLM("error")).decide(ctx).fallback


# ------------------------------------------------------------------ multi agent
def test_unanimous_specialists_skip_the_coordinator(ctx):
    sid = ctx.candidates[0].shot_id
    llm = Scripted({"offense": sid, "position": sid, "safety": sid})
    d = MultiAgent(llm).decide(ctx)
    assert d.shot_id == sid and d.trace["unanimous"] and "coordinator" not in llm.calls
    assert sorted(llm.calls) == ["offense", "position", "safety"] and d.usage["calls"] == 3


def test_coordinator_chooses_from_the_union(ctx):
    a, b, c = (x.shot_id for x in ctx.candidates[:3])
    llm = Scripted({"offense": a, "position": b, "safety": c}, coord=b)
    d = MultiAgent(llm).decide(ctx)
    assert d.shot_id == b and not d.trace["unanimous"] and "coordinator" in llm.calls
    assert "position" in d.trace["followed"]
    assert set(d.trace["shortlists"]) == {"offense", "position", "safety"}


def test_coordinator_outside_union_falls_back(ctx):
    a = ctx.candidates[0].shot_id
    outside = next(x.shot_id for x in ctx.candidates
                   if x.shot_id not in {y.shot_id for y in ctx.candidates[:3]})
    # every specialist nominates only the first three rows, the coordinator picks another
    llm = Scripted({"offense": a, "position": ctx.candidates[1].shot_id,
                    "safety": ctx.candidates[2].shot_id}, coord=outside)
    d = MultiAgent(llm).decide(ctx)
    assert llm.calls.count("coordinator") == 2  # one corrective retry, then the fallback rule
    assert d.fallback


def test_clean_picks_filters_dedupes_and_trims():
    sl = Shortlist(picks=[Pick(shot_id="S01", reason=" ".join(["w"] * 30)),
                          Pick(shot_id="S99", reason="invalid"), Pick(shot_id="S01", reason="dup"),
                          Pick(shot_id="S02", reason="ok"), Pick(shot_id="S03", reason="ok"),
                          Pick(shot_id="S04", reason="fourth")])
    out = clean_picks(sl, {"S01", "S02", "S03", "S04"})
    assert [p["shot_id"] for p in out] == ["S01", "S02", "S03"]
    assert len(out[0]["reason"].split()) == 20


def test_registry_builds_v2_policies():
    for name in ("stat", "b1", "sa", "ma"):
        assert hasattr(make_policy(name, llm="fake"), "decide")
    with pytest.raises(ValueError):
        make_policy("sa-vote")


# ------------------------------------------------------------------ ollama client (mocked)
def test_ollama_client_parses_caches_retries_and_reports_errors(tmp_path, monkeypatch):
    import json as _json

    import httpx

    sent = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": _json.dumps({"shot_id": "S02", "rationale": "x"})},
                    "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient("qwen-test", cache=ResponseCache(tmp_path / "c.sqlite"))
    r = client.complete(system="s", user="u", schema=SingleDecision, agent="single", repeat_idx=2)
    assert r.parsed.shot_id == "S02" and r.usage["input_tokens"] == 900
    assert sent["body"]["options"]["seed"] == 1002 and sent["body"]["format"]["properties"]["shot_id"]
    assert client.complete(system="s", user="u", schema=SingleDecision, agent="single",
                           repeat_idx=2).cached

    monkeypatch.setattr(OllamaClient, "RETRY_DELAYS_S", (0, 0))
    calls = {"n": 0}

    def flaky(url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("restarting")
        if calls["n"] == 2:
            raise httpx.ReadError("connection reset")
        return Resp()

    monkeypatch.setattr(httpx, "post", flaky)
    assert OllamaClient("q").complete(system="s", user="y", schema=SingleDecision,
                                      agent="a").parsed is not None and calls["n"] == 3

    def refuse(url, json=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    bad = OllamaClient("q").complete(system="s", user="z", schema=SingleDecision, agent="a")
    assert bad.parsed is None and "ollama serve" in bad.error


def test_offline_cache_only_mode(tmp_path):
    c = OllamaClient("q", cache=ResponseCache(tmp_path / "c.sqlite"), offline=True)
    r = c.complete(system="s", user="u", schema=SingleDecision, agent="a")
    assert r.parsed is None and "offline" in r.error
