import numpy as np
import pytest

from cuemind.sim.execute import Stroke, extract_events, set_stroke, simulate
from cuemind.sim.noise import NOISE_PRESETS, get_noise, rng_for
from cuemind.sim.state import TableState, positions_from_system, to_system

# Ball 1 sits on the diagonal towards the top-left pocket; the cue ball is behind it.
STRAIGHT = TableState(cue=(0.40, 1.40), balls={"1": (0.20, 1.70)})


def _phi_to(a, b):
    return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))


def test_straight_pot_and_input_untouched():
    system = to_system(STRAIGHT)
    before = positions_from_system(system)
    shot = simulate(system, Stroke(phi=_phi_to(STRAIGHT.cue, STRAIGHT.balls["1"]), V0=2.0))
    ev = extract_events(shot)
    assert ev.first_ball_hit == "1"
    assert ev.object_pocketed == {"1": "lt"}
    assert not ev.scratched
    assert positions_from_system(system) == before  # simulate() works on a copy


def test_simulation_is_deterministic():
    system = to_system(STRAIGHT)
    s = Stroke(phi=_phi_to(STRAIGHT.cue, STRAIGHT.balls["1"]) + 3, V0=3.0, b=-0.2)
    assert positions_from_system(simulate(system, s)) == positions_from_system(simulate(system, s))


def test_stroke_parameters_are_always_set():
    """Regression: pooltool's Cue defaults to b=0.25 (follow); we must override it."""
    system = to_system(STRAIGHT)
    set_stroke(system, Stroke(phi=10, V0=1))
    assert system.cue.b == 0.0 and system.cue.a == 0.0


def test_noise_common_random_numbers():
    nm = get_noise("medium")
    s = Stroke(phi=45.0, V0=2.5, b=0.1)
    assert nm.apply(s, rng_for(7, 3, "execute")) == nm.apply(s, rng_for(7, 3, "execute"))
    assert nm.apply(s, rng_for(7, 3, "execute")) != nm.apply(s, rng_for(7, 4, "execute"))
    assert get_noise("none").apply(s, rng_for(1)) == s.clipped()


def test_noise_clips_and_presets():
    big = NOISE_PRESETS["high"]
    for i in range(50):
        out = big.apply(Stroke(phi=0, V0=8.4, b=0.49), rng_for(i))
        assert 0.3 <= out.V0 <= 8.5 and -0.5 <= out.b <= 0.5 and 0 <= out.phi < 360
    with pytest.raises(ValueError):
        get_noise("extreme")
