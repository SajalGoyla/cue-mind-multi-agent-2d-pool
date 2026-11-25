"""v2 menu rule (D-029): all variants forecast, top 15 kept, every legal ball covered,
geometry-only descriptions, seeded shuffle."""

import pytest

from cuemind.candidates.generate import generate
from cuemind.candidates.model import candidates_hash
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.policies.base import DecisionContext
from cuemind.policies.search_bot import SearchBot, SearchWeights, expected_value
from cuemind.sim.state import TableState

OPEN = TableState(cue=(0.5, 0.4), balls={"1": (0.25, 1.0), "2": (0.7, 1.5), "3": (0.45, 1.75),
                                          "9": (0.5, 1.2), "12": (0.8, 0.8), "8": (0.2, 1.5)},
                  groups={"P1": "solids", "P2": "stripes"})


@pytest.fixture(scope="module")
def menu():
    return prepare(OPEN, PrepConfig(), rollout_key=(1, 7))


def test_menu_size_ids_and_forecasts(menu):
    assert 2 <= len(menu) <= 15
    assert [c.shot_id for c in menu] == [f"S{i + 1:02d}" for i in range(len(menu))]
    assert sum(c.shot_type == "pot" for c in menu) <= 12
    assert sum(c.shot_type == "safety" for c in menu) <= 3
    assert all(c.rollouts is not None and c.rollouts.n == 3 for c in menu)


def test_every_legal_ball_with_a_pot_is_represented(menu):
    pottable = {c.target_ball for c in generate(OPEN) if c.shot_type == "pot"}
    assert pottable and pottable <= {c.target_ball for c in menu if c.shot_type == "pot"}


def test_all_speed_spin_variants_are_generated():
    pots = [c for c in generate(OPEN) if c.shot_type == "pot"]
    first = (pots[0].target_ball, pots[0].pocket)
    variants = {(c.power_label, c.spin_label) for c in pots if (c.target_ball, c.pocket) == first}
    assert len(variants) == 9


def test_descriptions_are_geometry_only(menu):
    for c in menu:
        assert "%" not in c.description and "realistic" not in c.description
        assert "preview" not in c.description


def test_menu_is_deterministic_and_order_is_shuffled(menu):
    again = prepare(OPEN, PrepConfig(), rollout_key=(1, 7))
    assert candidates_hash(again) == candidates_hash(menu)
    w = SearchWeights()
    evs = [expected_value(c, w) for c in menu]
    assert evs != sorted(evs, reverse=True)  # not listed best-first


def test_stat_picks_the_best_forecast(menu):
    d = SearchBot().decide(DecisionContext(state=OPEN, candidates=menu))
    w = SearchWeights()
    assert abs(d.trace["ev"][d.shot_id] - max(expected_value(c, w) for c in menu)) < 1e-9
