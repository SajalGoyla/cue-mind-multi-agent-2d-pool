"""v3 conditions: simulation-free hints + geometric menu (A) and forecasts for everyone (B)."""

from __future__ import annotations

import pytest

from cuemind.candidates.hints import Hint, hint_text, rule_score
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.game.runner import play_game
from cuemind.llm import promptlib as P
from cuemind.policies.base import DecisionContext
from cuemind.policies.registry import make_policy
from cuemind.policies.rule_bot import RuleBot
from cuemind.sim.execute import warm_up
from cuemind.sim.state import TableState

A = PrepConfig(menu="geometric", info="hints")
B = PrepConfig(menu="forecast", info="forecasts", n_forecast=1)


@pytest.fixture(scope="module")
def state() -> TableState:
    warm_up()
    return TableState(cue=(0.5, 0.5), shooter="P1", groups={"P1": "solids", "P2": "stripes"},
                      balls={"1": (0.5, 1.2), "2": (0.25, 1.5), "3": (0.75, 0.9), "9": (0.3, 0.9),
                             "10": (0.7, 1.6), "8": (0.5, 1.75)})


def test_prep_config_validates():
    with pytest.raises(ValueError):
        PrepConfig(menu="geometric", info="forecasts")
    with pytest.raises(ValueError):
        PrepConfig(menu="forecast", info="hints")
    assert PrepConfig().info == "geometry"


def test_geometric_menu_uses_no_simulation(state):
    menu = prepare(state, A, (1, 1))
    assert 2 <= len(menu) <= 15
    assert all(c.rollouts is None for c in menu)          # no forecast anywhere
    pots = [c for c in menu if c.shot_type == "pot"]
    assert pots and all(c.hint is not None for c in pots)
    assert {c.target_ball for c in pots} <= {"1", "2", "3"}
    assert len({c.target_ball for c in pots}) >= 2          # several legal balls represented
    assert all("cue ball" in c.description for c in pots)
    assert [c.shot_id for c in menu] == [f"S{i:02d}" for i in range(1, len(menu) + 1)]


def test_geometric_menu_is_deterministic(state):
    a = [c.key for c in prepare(state, A, (1, 1))]
    assert a == [c.key for c in prepare(state, A, (1, 1))]


def test_forecast_info_shows_statistics(state):
    menu = prepare(state, B, (1, 1))
    assert all(c.rollouts is not None for c in menu)
    assert all("realistic strokes" in c.description for c in menu)


def test_hint_text_and_rule_score():
    h = Hint(direction=(1.0, 0.0), travel_m=0.8, end=(0.5, 1.0), scratch_pocket=None,
             next_ease=0.8, next_ball="3", cushions=1)
    assert "easy next shot on the 3" in hint_text(h)
    warn = Hint(direction=(1.0, 0.0), travel_m=0.8, end=(0.5, 1.0), scratch_pocket="rt",
                next_ease=0.0, next_ball=None, cushions=0)
    assert "WARNING" in hint_text(warn)


def test_rule_bot_avoids_scratch_warnings(state):
    menu = prepare(state, A, (1, 1))
    d = RuleBot().decide(DecisionContext(state=state, candidates=menu, info="hints"))
    c = next(c for c in menu if c.shot_id == d.shot_id)
    best = max(rule_score(x, Hint(**x.hint) if x.hint else None) for x in menu)
    assert rule_score(c, Hint(**c.hint) if c.hint else None) == best or c.shot_type == "safety"


def test_prompts_v2_unchanged_and_info_blocks():
    assert "Physics hints" not in P.system_single()
    assert P.system_single("geometry") == P.system_single()
    assert "Physics hints" in P.system_single("hints")
    assert "Forecast statistics" in P.system_coordinator("forecasts")
    assert "Physics hints" in P.system_specialist("safety", "hints")


@pytest.mark.parametrize("prep,ref", [(A, "rule"), (B, "stat")])
def test_full_game_each_condition(prep, ref):
    pol = {"P1": make_policy(ref), "P2": make_policy("ma", llm="fake")}
    res = play_game(pol, seed=3, noise="none", prep=prep, max_shots=40)
    assert res.status in ("finished", "shot_cap"), res.error
    decs = [r for r in res.records if r["kind"] == "decision"]
    assert decs and all(r["prep"]["info"] == prep.info for r in decs)
