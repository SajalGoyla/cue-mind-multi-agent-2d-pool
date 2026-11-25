"""The raw menu (what every policy sees) and B's private simulation."""

from cuemind.candidates.annotate import rollouts
from cuemind.candidates.describe import diamonds, state_summary
from cuemind.candidates.generate import generate
from cuemind.candidates.model import candidates_hash
from cuemind.candidates.pipeline import MenuConfig, build_menu
from cuemind.policies.base import DecisionContext, fallback_choice
from cuemind.policies.search_bot import SearchBot
from cuemind.sim.state import TABLE_L, TABLE_W, TableState

MID = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "4": (0.3, 0.9),
                                         "10": (0.5, 1.0), "12": (0.8, 1.7), "8": (0.3, 1.3)},
                 groups={"P1": "solids", "P2": "stripes"})
# Ball 1 hangs in front of the top-left pocket: a near-certain pot.
HANGER = TableState(cue=(0.40, 1.40), balls={"1": (0.12, 1.86), "9": (0.8, 0.4), "8": (0.7, 1.0)},
                    groups={"P1": "solids", "P2": "stripes"})


def _targets(cands):
    out = []
    for c in cands:
        if c.shot_type == "pot" and (c.target_ball, c.pocket) not in out:
            out.append((c.target_ball, c.pocket))
    return out


def test_menu_rule_and_ids():
    menu = build_menu(MID, menu_key=(1, 2))
    pots = [c for c in menu if c.shot_type == "pot"]
    assert 1 <= len(menu) <= 15
    assert sorted(c.shot_id for c in menu) == [f"S{i + 1:02d}" for i in range(len(menu))]
    assert sum(c.shot_type == "safety" for c in menu) <= 3
    assert all(c.power_label == "medium" for c in pots)
    kept = set(_targets(pots))
    assert kept == set(_targets(generate(MID))[:4])  # the 4 geometrically easiest pairs
    for t in kept:  # each kept pair offers every spin
        assert {c.spin_label for c in pots if (c.target_ball, c.pocket) == t} == {"draw", "stun", "follow"}


def test_menu_is_deterministic_and_shuffled_by_key():
    a, b = build_menu(MID, menu_key=(1, 2)), build_menu(MID, menu_key=(1, 2))
    assert candidates_hash(a) == candidates_hash(b)
    orders = {tuple(c.key for c in build_menu(MID, menu_key=(1, k))) for k in range(5)}
    assert len(orders) > 1
    assert {c.key for c in build_menu(MID, MenuConfig(shuffle=False))} == {c.key for c in a}


def test_llm_text_has_no_simulated_outcomes():
    for c in build_menu(MID, menu_key=(0,)):
        assert c.rollouts is None
        for word in ("%", "strokes", "cue ends", "opponent's chance"):
            assert word not in c.description
    assert diamonds((TABLE_W, TABLE_L)) == (4.0, 8.0)
    s = state_summary(MID)
    assert "You play solids" in s and "Cue ball at" in s
    eight = TableState(cue=(0.5, 0.5), balls={"8": (0.5, 1.5), "9": (0.2, 0.2)},
                       groups={"P1": "solids", "P2": "stripes"})
    text = state_summary(eight)
    assert "ON THE 8" in text and "any other foul gives the opponent ball in hand" in text


def test_rollout_statistics_are_valid_and_order_independent():
    cands = generate(MID)[:5]
    fwd = rollouts(MID, cands, 4, "medium", (1,))
    rev = rollouts(MID, list(reversed(cands)), 4, "medium", (1,))
    assert [c.rollouts for c in fwd] == [c.rollouts for c in reversed(rev)]
    for c in fwd:
        r = c.rollouts
        for v in (r.p_pot, r.p_scratch, r.p_foul, r.p_keep_turn, r.next_ease, r.opp_threat):
            assert 0.0 <= v <= 1.0
        assert r.p_scratch <= r.p_foul


def test_hanging_ball_pots_in_a_noise_free_simulation():
    sims = rollouts(HANGER, build_menu(HANGER, menu_key=(0,)), 1, "none", (0,))
    pots_1 = [c for c in sims if c.shot_type == "pot" and c.target_ball == "1" and c.pocket == "lt"]
    assert pots_1 and max(c.rollouts.p_pot for c in pots_1) == 1.0


def test_b_simulates_privately_and_picks_the_highest_value():
    menu = build_menu(MID, menu_key=(3,))
    d = SearchBot().decide(DecisionContext(state=MID, candidates=menu, noise="none", sim_key=(3,)))
    assert d.trace["n_rollouts"] == 1  # no noise: one run is exact
    assert d.trace["ev"][d.shot_id] == max(d.trace["ev"].values())
    assert all(c.rollouts is None for c in menu)  # the shared menu is untouched
    d2 = SearchBot(n_rollouts=3).decide(DecisionContext(state=MID, candidates=menu, noise="low",
                                                        sim_key=(3,)))
    assert d2.trace["n_rollouts"] == 3


def test_fallback_prefers_the_easiest_pot():
    menu = build_menu(MID, menu_key=(0,))
    fb = fallback_choice(menu)
    assert fb.shot_type == "pot"
    assert fb.geometry.ease == max(c.geometry.ease for c in menu if c.shot_type == "pot")
