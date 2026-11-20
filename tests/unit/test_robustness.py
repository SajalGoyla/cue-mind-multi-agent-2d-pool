"""Regression tests for problems found while running games (see decisions D-013, D-014)."""

import os
import subprocess
import sys

import pooltool as pt
import pytest

from cuemind.candidates.annotate import rollouts
from cuemind.candidates.generate import generate
from cuemind.sim import execute
from cuemind.sim.execute import SimulationError, Stroke, simulate
from cuemind.sim.state import TableState, to_system

CODE = "from cuemind.sim.state import rack_state, state_hash; print(state_hash(rack_state(119)))"
MID = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "3": (0.75, 1.2), "10": (0.5, 1.0)},
                 groups={"P1": "solids", "P2": "stripes"})


@pytest.mark.slow
def test_rack_is_identical_across_python_hash_seeds():
    """D-013: pooltool assigns rack numbers via set order, which depends on PYTHONHASHSEED."""
    hashes = set()
    for h in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": h, "PYTHONUTF8": "1"}
        out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True,
                             text=True, check=True)
        hashes.add(out.stdout.strip().splitlines()[-1])
    assert len(hashes) == 1


def test_simulate_retries_with_a_nudge(monkeypatch):
    real = pt.simulate
    calls = {"n": 0}

    def flaky(system, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise AssertionError("pooltool internal")
        return real(system, **kw)

    monkeypatch.setattr(execute.pt, "simulate", flaky)
    shot = simulate(to_system(MID), Stroke(phi=90, V0=2))
    assert calls["n"] == 2 and shot is not None


def test_simulate_gives_up_with_simulation_error(monkeypatch):
    def broken(system, **kw):
        raise AssertionError("always")

    monkeypatch.setattr(execute.pt, "simulate", broken)
    with pytest.raises(SimulationError):
        simulate(to_system(MID), Stroke(phi=90, V0=2))


def test_failed_rollouts_count_as_fouls(monkeypatch):
    cands = [c for c in generate(MID) if c.shot_type == "pot"][:1]

    def broken(system, **kw):
        raise AssertionError("always")

    monkeypatch.setattr(execute.pt, "simulate", broken)
    r = rollouts(MID, cands, 3, "medium", (0,))[0].rollouts
    assert r.p_foul == 1.0 and r.p_pot == 0.0 and r.opp_threat == 1.0
