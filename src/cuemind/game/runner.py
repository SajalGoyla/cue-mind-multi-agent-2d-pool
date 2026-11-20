"""Play a full game between two policies, logging every decision as one JSON record.

Usage (baseline sanity check):
    python -m cuemind.game.runner --p1 b1 --p2 b0 --games 2 --noise medium
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cuemind.candidates.model import candidates_hash
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.game.ball_in_hand import resolve_ball_in_hand
from cuemind.game.break_shot import break_stroke
from cuemind.game.rules import Call, advance, judge
from cuemind.policies.base import Decision, DecisionContext, Policy, fallback_choice
from cuemind.sim.execute import extract_events, simulate, warm_up
from cuemind.sim.noise import get_noise, rng_for
from cuemind.sim.state import TableState, positions_from_system, rack_state, state_hash, to_system

HISTORY_LEN = 3


@dataclass
class GameResult:
    game_id: str
    seed: int
    players: dict[str, str]          # seat -> policy name
    status: str                      # finished | shot_cap | error
    winner: str | None               # seat, None for a draw/error
    n_shots: int
    wall_s: float
    fouls: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    records: list[dict] = field(default_factory=list)

    def score(self, seat: str) -> float:
        if self.winner is None:
            return 0.5
        return 1.0 if self.winner == seat else 0.0

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "records"}


def _call_for(cand) -> Call:
    if cand.shot_type == "pot":
        return Call(ball=cand.target_ball, pocket=cand.pocket)
    return Call(ball=cand.target_ball, pocket=None, safety=True)


def _decide_safely(policy: Policy, ctx: DecisionContext) -> Decision:
    """A policy may never crash the game: exceptions and unknown ids fall back."""
    try:
        d = policy.decide(ctx)
        if ctx.by_id(d.shot_id) is not None:
            return d
        err = f"unknown shot_id {d.shot_id!r}"
        trace, usage = d.trace, d.usage
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        trace, usage = {"traceback": traceback.format_exc(limit=3)}, {}
    fb = fallback_choice(ctx.candidates)
    return Decision(shot_id=fb.shot_id, intent="offense" if fb.shot_type == "pot" else "safety",
                    rationale=f"fallback ({err})", fallback=True,
                    trace={**trace, "fallback_reason": err}, usage=usage)


def play_game(policies: dict[str, Policy], seed: int, *, noise: str = "medium",
              prep: PrepConfig | None = None, start_state: TableState | None = None,
              max_shots: int = 150, game_id: str | None = None,
              log_path: Path | None = None, exp_id: str = "adhoc") -> GameResult:
    """Play one game. `policies` maps seat ("P1", "P2") to a Policy. P1 breaks unless a
    start_state is given. Execution noise uses common random numbers keyed by
    (seed, shot index), so paired games with swapped seats share the same noise draws."""
    warm_up()
    prep = prep or PrepConfig(noise=noise)
    noise_model = get_noise(noise)
    game_id = game_id or f"{exp_id}_seed{seed}"
    state = start_state if start_state is not None else rack_state(seed)
    history: list[dict] = []
    records: list[dict] = []
    fouls: Counter = Counter()
    t_game = time.perf_counter()
    status, winner, error = "shot_cap", None, None
    players = {seat: p.name for seat, p in policies.items()}

    try:
        for k in range(max_shots):
            t0 = time.perf_counter()
            seat = state.shooter
            rec: dict[str, Any] = {"exp_id": exp_id, "game_id": game_id, "seed": seed,
                                   "shot_index": k, "seat": seat, "system": players[seat]}
            if state.break_shot:
                stroke = break_stroke(state, rng_for(seed, "break"))
                call = Call(ball=None, pocket=None, safety=True)
                rec.update(kind="break", state_hash=state_hash(state), state=state.model_dump())
                executed = stroke
            else:
                rec["ball_in_hand"] = state.ball_in_hand or state.cue is None
                state = resolve_ball_in_hand(state)
                cands = prepare(state, prep, rollout_key=(seed, k))
                ctx = DecisionContext(state=state, candidates=cands, history=history[-HISTORY_LEN:],
                                      game_id=game_id, rng=rng_for(seed, k, "policy", seat))
                t_dec = time.perf_counter()
                decision = _decide_safely(policies[seat], ctx)
                decision_s = time.perf_counter() - t_dec
                cand = ctx.by_id(decision.shot_id)
                call = _call_for(cand)
                executed = noise_model.apply(cand.stroke, rng_for(seed, k, "execute"))
                rec.update(
                    kind="decision", state_hash=state_hash(state), state=state.model_dump(),
                    prep=prep.to_dict(), candidates_hash=candidates_hash(cands),
                    candidates=[c.model_dump() for c in cands],
                    decision={"shot_id": decision.shot_id, "intent": decision.intent,
                              "rationale": decision.rationale, "confidence": decision.confidence},
                    fallback=decision.fallback, trace=decision.trace, usage=decision.usage,
                    planned=cand.stroke.to_dict(), decision_latency_s=round(decision_s, 3),
                )
            shot = simulate(to_system(state), executed)
            ev = extract_events(shot)
            j = judge(state, ev, call)
            cue_after, balls_after = positions_from_system(shot)
            if j.foul:
                fouls[j.foul.split(" (")[0]] += 1
            rec.update(
                executed=executed.to_dict(),
                outcome={"potted": j.potted, "legal": j.legal, "foul": j.foul,
                         "turn_over": j.turn_over, "game_over": j.game_over, "winner": j.winner,
                         "called_ball_made": j.called_ball_made,
                         "group_assigned": j.group_for_shooter, "note": j.note,
                         "first_ball_hit": ev.first_ball_hit, "truncated": ev.truncated},
                step_s=round(time.perf_counter() - t0, 3),
            )
            records.append(rec)
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            if rec["kind"] == "decision":
                history.append({"seat": seat, "shot_id": rec["decision"]["shot_id"],
                                "intent": rec["decision"]["intent"],
                                "result": ("foul: " + j.foul) if j.foul else
                                ("potted " + ",".join(j.potted) if j.potted else "nothing potted")})
            if j.game_over:
                status, winner = "finished", j.winner
                break
            state = advance(state, j, cue_after, balls_after)
    except Exception as exc:  # noqa: BLE001 - record and surface, never hang a tournament
        status, error = "error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"

    return GameResult(game_id=game_id, seed=seed, players=players, status=status, winner=winner,
                      n_shots=len(records), wall_s=round(time.perf_counter() - t_game, 2),
                      fouls=dict(fouls), error=error, records=records)


def main() -> None:
    from cuemind.policies.registry import make_policy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1", default="b1")
    ap.add_argument("--p2", default="b1")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", default="medium")
    ap.add_argument("--access", default="L2")
    ap.add_argument("--log", default=None, help="JSONL path for per-shot records")
    a = ap.parse_args()
    prep = PrepConfig(access_level=a.access, noise=a.noise)
    for g in range(a.games):
        res = play_game({"P1": make_policy(a.p1), "P2": make_policy(a.p2)}, seed=a.seed + g,
                        noise=a.noise, prep=prep, log_path=Path(a.log) if a.log else None)
        print(json.dumps(res.summary()))


if __name__ == "__main__":
    main()
