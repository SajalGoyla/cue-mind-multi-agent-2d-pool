import math

from cuemind.sim.state import (BALL_R, TableState, positions_from_system, rack_state, state_hash,
                               to_system, validate)


def test_rack_is_deterministic_per_seed():
    assert state_hash(rack_state(3)) == state_hash(rack_state(3))
    assert state_hash(rack_state(3)) != state_hash(rack_state(4))
    r = rack_state(0)
    assert len(r.balls) == 15 and r.break_shot and validate(r) == []


def test_validate_catches_bad_positions():
    s = TableState(cue=(0.5, 0.5), balls={"1": (0.5, 0.5 + BALL_R)})
    assert any("overlap" in p for p in validate(s))
    s = TableState(cue=(0.5, 0.5), balls={"1": (5.0, 0.5)})
    assert any("off the playing surface" in p for p in validate(s))
    s = TableState(cue=(0.5, 0.5), balls={"1": (math.nan, 0.5)})
    assert any("non-finite" in p for p in validate(s))
    s = TableState(cue=None, balls={"1": (0.3, 0.3)})
    assert any("ball_in_hand" in p for p in validate(s))
    s = TableState(cue=(0.5, 0.5), balls={"1": (0.3, 0.3)}, groups={"P1": "solids", "P2": "solids"})
    assert any("same group" in p for p in validate(s))


def test_legal_targets_by_group():
    balls = {"1": (0.2, 0.2), "9": (0.4, 0.4), "8": (0.6, 0.6)}
    open_t = TableState(cue=(0.8, 1.5), balls=balls)
    assert open_t.legal_targets == ["1", "9"]
    s = TableState(cue=(0.8, 1.5), balls=balls, groups={"P1": "solids", "P2": "stripes"})
    assert s.legal_targets == ["1"]
    assert s.model_copy(update={"shooter": "P2"}).legal_targets == ["9"]
    s3 = TableState(cue=(0.8, 1.5), balls={"9": (0.4, 0.4), "8": (0.6, 0.6)},
                    groups={"P1": "solids", "P2": "stripes"})
    assert s3.on_the_eight("P1") and s3.legal_targets == ["8"]
    assert not s3.on_the_eight("P2")


def test_system_round_trip():
    s = TableState(cue=(0.3, 0.4), balls={"1": (0.5, 1.0), "12": (0.7, 1.6)})
    cue, balls = positions_from_system(to_system(s))
    assert cue == s.cue and balls == s.balls
