"""Execute a cue stroke in pooltool and extract the rule-relevant facts from the events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pooltool as pt
from pooltool.events import EventType, by_ball, by_type, filter_events

MAX_EVENTS = 3000  # guard against pathological, very long simulations (breaks, clusters)
V0_MIN, V0_MAX = 0.3, 8.5
TIP_MAX = 0.5


@dataclass(frozen=True)
class Stroke:
    """A cue stroke in pooltool's parameterisation.

    phi: aim direction in degrees, counter-clockwise from +x.
    V0: cue-stick speed in m/s.
    a, b: tip offsets (side spin, top/back spin) in units of the ball radius.
    theta: cue elevation in degrees.
    """

    phi: float
    V0: float
    a: float = 0.0
    b: float = 0.0
    theta: float = 0.0

    def clipped(self) -> "Stroke":
        return Stroke(phi=self.phi % 360.0, V0=min(max(self.V0, V0_MIN), V0_MAX),
                      a=min(max(self.a, -TIP_MAX), TIP_MAX), b=min(max(self.b, -TIP_MAX), TIP_MAX),
                      theta=self.theta)

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class ShotEvents:
    """Rule-relevant summary of one simulated shot."""

    first_ball_hit: str | None
    pocketed: dict[str, str] = field(default_factory=dict)  # ball id -> pocket id (incl. "cue")
    cushion_after_contact: bool = False
    object_balls_hit_cushion: int = 0
    n_events: int = 0
    truncated: bool = False

    @property
    def scratched(self) -> bool:
        return "cue" in self.pocketed

    @property
    def object_pocketed(self) -> dict[str, str]:
        return {b: p for b, p in self.pocketed.items() if b != "cue"}


def set_stroke(system: pt.System, stroke: Stroke) -> None:
    s = stroke.clipped()
    # Always pass every parameter: pooltool's Cue default has b=0.25 (follow).
    system.cue.set_state(V0=s.V0, phi=s.phi, theta=s.theta, a=s.a, b=s.b, cue_ball_id="cue")


class SimulationError(RuntimeError):
    """pooltool failed on this stroke even after the nudged retries."""


NUDGES_DEG = (0.0, 1e-4, -1e-4, 3e-4)


def simulate(system: pt.System, stroke: Stroke, *, continuous: bool = False,
             dt: float = 0.02) -> pt.System:
    """Simulate a stroke on a copy of `system` and return the simulated copy (the input is
    never modified).

    pooltool very occasionally raises an internal AssertionError on degenerate collision
    geometry (seen once in 20 games). We retry with the aim nudged by a negligible angle
    (1e-4 degrees) before giving up with SimulationError (D-014).
    """
    kwargs = {"inplace": True, "max_events": MAX_EVENTS}
    if continuous:
        kwargs.update(continuous=True, dt=dt)
    last_exc: Exception | None = None
    for nudge in NUDGES_DEG:
        shot = system.copy()
        shot.reset_history()
        s = stroke if nudge == 0.0 else Stroke(stroke.phi + nudge, stroke.V0, stroke.a, stroke.b,
                                                stroke.theta)
        set_stroke(shot, s)
        try:
            pt.simulate(shot, **kwargs)
        except Exception as exc:  # noqa: BLE001 - pooltool internal failure
            last_exc = exc
            continue
        return shot
    raise SimulationError(f"pooltool failed for {stroke}: {last_exc!r}") from last_exc


def extract_events(shot: pt.System) -> ShotEvents:
    """Summarise a simulated shot. Uses exact event ids (avoids pooltool's substring
    `is_ball_pocketed` quirk, where "1" also matches "11")."""
    events = shot.events
    contacts = filter_events(events, by_ball("cue"), by_type(EventType.BALL_BALL))
    first = None
    t_first = None
    if contacts:
        ev = contacts[0]
        first = next(i for i in ev.ids if i != "cue")
        t_first = ev.time

    pocketed: dict[str, str] = {}
    for ev in filter_events(events, by_type(EventType.BALL_POCKET)):
        ball_id, pocket_id = ev.ids
        pocketed.setdefault(ball_id, pocket_id)

    cushion_types = [EventType.BALL_LINEAR_CUSHION, EventType.BALL_CIRCULAR_CUSHION]
    cushion_events = filter_events(events, by_type(cushion_types))
    cushion_after = t_first is not None and any(ev.time >= t_first for ev in cushion_events)
    obj_cushion = {ev.ids[0] for ev in cushion_events if ev.ids[0] != "cue"}
    return ShotEvents(
        first_ball_hit=first,
        pocketed=pocketed,
        cushion_after_contact=bool(cushion_after),
        object_balls_hit_cushion=len(obj_cushion),
        n_events=len(events),
        truncated=len(events) >= MAX_EVENTS,
    )


_warmed = False


def warm_up() -> None:
    """Trigger Numba JIT compilation once per process (first call can take ~2-30 s)."""
    global _warmed
    if not _warmed:
        pt.simulate(pt.System.example())
        _warmed = True
