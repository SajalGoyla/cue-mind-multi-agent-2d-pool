import pytest

from cuemind.candidates.annotate import preview, rollouts
from cuemind.candidates.describe import describe, diamonds, state_summary
from cuemind.candidates.generate import generate
from cuemind.candidates.model import candidates_hash
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.policies.base import DecisionContext, fallback_choice
from cuemind.policies.search_bot import SearchBot, SearchWeights, expected_value
from cuemind.sim.state import TABLE_L, TABLE_W, TableState

MID = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "4": (0.3, 0.9),
                                         "10": (0.5, 1.0), "12": (0.8, 1.7), "8": (0.3, 1.3)},
                 groups={"P1": "solids", "P2": "stripes"})
# Ball 1 hangs in front of the top-left pocket: a near-certain pot.
HANGER = TableState(cue=(0.40, 1.40), balls={"1": (0.12, 1.86), "9": (0.8, 0.4), "8": (0.7, 1.0)},
                    groups={"P1": "solids", "P2": "stripes"})


@pytest.fixture(scope="module")
def menu_l2():
    return prepare(MID, PrepConfig(access_level="L2", n_rollouts=6), rollout_key=(11, 3))


def test_rollout_statistics_are_valid_probabilities(menu_l2):
    for c in menu_l2:
        r = c.rollouts
        assert r is not None and r.n == 6
        for v in (r.p_pot, r.p_scratch, r.p_foul, r.p_keep_turn, r.next_ease, r.opp_threat):
            assert 0.0 <= v <= 1.0
        assert r.p_scratch <= r.p_foul
        if c.shot_type == "safety":
            assert r.p_pot == 0.0 and r.p_keep_turn == 0.0


def test_l2_is_deterministic_and_order_independent(menu_l2):
    again = prepare(MID, PrepConfig(access_level="L2", n_rollouts=6), rollout_key=(11, 3))
    assert candidates_hash(again) == candidates_hash(menu_l2)
    cands = generate(MID)[:5]
    fwd = rollouts(MID, cands, 4, "medium", (1,))
    rev = rollouts(MID, list(reversed(cands)), 4, "medium", (1,))
    assert [c.rollouts for c in fwd] == [c.rollouts for c in reversed(rev)]


def test_hanging_ball_is_a_near_certain_pot():
    menu = prepare(HANGER, PrepConfig(access_level="L2", n_rollouts=8), rollout_key=(0,))
    best = max((c for c in menu if c.shot_type == "pot"), key=lambda c: c.rollouts.p_pot)
    assert best.target_ball == "1" and best.pocket == "lt" and best.rollouts.p_pot >= 0.75


def test_preview_matches_a_noise_free_pot():
    pv = preview(HANGER, [c for c in generate(HANGER) if c.pocket == "lt"][:1])[0].preview
    assert pv.pots_called and pv.legal and not pv.scratch


def test_menu_size_ids_and_menu_identical_across_levels_rules(menu_l2):
    assert 2 <= len(menu_l2) <= 15
    assert [c.shot_id for c in menu_l2] == [f"S{i + 1:02d}" for i in range(len(menu_l2))]
    assert sum(c.shot_type == "safety" for c in menu_l2) <= 3
    assert all(c.rollouts.p_pot >= 0.10 for c in menu_l2 if c.shot_type == "pot")
    l0 = prepare(MID, PrepConfig(access_level="L0"))
    assert all(c.rollouts is None and c.preview is None for c in l0)
    with pytest.raises(ValueError):
        PrepConfig(access_level="L9")


def test_descriptions_respect_access_level(menu_l2):
    c = menu_l2[0]
    assert "realistic strokes" in c.description
    assert "realistic strokes" not in describe(c.model_copy(update={"rollouts": None}), "L0")
    assert diamonds((TABLE_W, TABLE_L)) == (4.0, 8.0)
    s = state_summary(MID)
    assert "You play solids" in s and "Cue ball at" in s
    eight = TableState(cue=(0.5, 0.5), balls={"8": (0.5, 1.5), "9": (0.2, 0.2)},
                       groups={"P1": "solids", "P2": "stripes"})
    assert "ON THE 8" in state_summary(eight)


def test_search_bot_prefers_value_and_fallback_prefers_pots(menu_l2):
    ctx = DecisionContext(state=MID, candidates=menu_l2)
    d = SearchBot().decide(ctx)
    best_ev = max(expected_value(c, SearchWeights()) for c in menu_l2)
    assert abs(d.trace["ev"][d.shot_id] - best_ev) < 1e-3
    fb = fallback_choice(menu_l2)
    assert fb.shot_type == "pot"
    assert fb.rollouts.p_pot == max(c.rollouts.p_pot for c in menu_l2 if c.shot_type == "pot")
