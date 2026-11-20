import math

from cuemind.candidates.generate import contact_fallback, generate
from cuemind.candidates.model import candidates_hash
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.game.ball_in_hand import place_cue_ball, resolve_ball_in_hand
from cuemind.game.rules import Call, judge
from cuemind.sim.execute import extract_events, simulate
from cuemind.sim.state import BALL_R, TABLE_L, TABLE_W, TableState, rack_state, to_system

MID = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "10": (0.5, 1.0),
                                         "8": (0.3, 1.3)}, groups={"P1": "solids", "P2": "stripes"})


def test_pot_candidates_found_and_deterministic():
    a, b = generate(MID), generate(MID)
    assert candidates_hash(a) == candidates_hash(b)
    pots = [c for c in a if c.shot_type == "pot"]
    assert pots and {c.target_ball for c in pots} <= {"1", "3"}
    assert len({(c.power_label, c.spin_label) for c in pots}) == 9


def test_safeties_are_legal_in_noise_free_simulation():
    system = to_system(MID)
    for c in [c for c in generate(MID) if c.shot_type == "safety"]:
        ev = extract_events(simulate(system, c.stroke))
        assert judge(MID, ev, Call(c.target_ball, None, True)).foul is None


def test_pruned_menu_is_small_diverse_and_relabelled():
    cands = prepare(MID, PrepConfig(access_level="L0"))
    assert 1 <= len(cands) <= 15
    assert [c.shot_id for c in cands] == [f"S{i + 1:02d}" for i in range(len(cands))]
    pots = [c for c in cands if c.shot_type == "pot"]
    n_geoms = len({(c.target_ball, c.pocket) for c in generate(MID) if c.shot_type == "pot"})
    # Round-robin pruning keeps every geometry when there are at most 12 of them.
    assert len({(c.target_ball, c.pocket) for c in pots}) == min(n_geoms, 12)
    open_table = TableState(cue=(0.5, 0.4), balls={"1": (0.25, 1.0), "2": (0.7, 1.5),
                                                   "9": (0.5, 1.2), "12": (0.8, 0.8)})
    many = prepare(open_table, PrepConfig(access_level="L0"))
    assert len({(c.target_ball, c.pocket) for c in many if c.shot_type == "pot"}) >= 3


def test_contact_fallback_targets_legal_balls():
    fb = contact_fallback(MID)
    assert {c.target_ball for c in fb} == {"1", "3"}
    assert all(c.shot_type == "contact_only" for c in fb)


def test_ball_in_hand_placement_is_free_and_deterministic():
    s = rack_state(1).model_copy(update={"cue": None, "ball_in_hand": True, "break_shot": False})
    spot1, _ = place_cue_ball(s)
    spot2, _ = place_cue_ball(s)
    assert spot1 == spot2
    assert all(math.dist(spot1, xy) >= 2 * BALL_R for xy in s.balls.values())
    assert BALL_R <= spot1[0] <= TABLE_W - BALL_R and BALL_R <= spot1[1] <= TABLE_L - BALL_R
    placed = resolve_ball_in_hand(s)
    assert placed.cue == spot1 and not placed.ball_in_hand
