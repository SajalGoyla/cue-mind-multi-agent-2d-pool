"""Positions benchmark: build (sample, annotate, tag, oracle) and evaluate decisions.

A position record stores the table state, one frozen fully-annotated menu (preview + L2
rollouts), difficulty and situation tags, and oracle values for every candidate. At run
time the same menu is *described* at the requested access level (L0 hides simulation
statistics, L1 shows only the preview), so systems and access levels are compared on
exactly the same shots and the same ground truth (D-019).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from cuemind.candidates.describe import describe_all
from cuemind.candidates.model import Candidate, candidates_hash
from cuemind.candidates.pipeline import PrepConfig, prepare
from cuemind.eval.oracle import OracleConfig, oracle_values
from cuemind.game.ball_in_hand import resolve_ball_in_hand
from cuemind.sim.state import (ALL_OBJECT_BALLS, BALL_R, SOLIDS, STRIPES, TABLE_L, TABLE_W,
                               TableState, state_hash, validate)

# ----------------------------------------------------------------------------- sampling


def sample_from_game(records: list[dict], rng: random.Random, k: int = 2) -> list[TableState]:
    """Pick up to k decision states (after cue placement) from one game's shot records."""
    states = [TableState.model_validate(r["state"]) for r in records if r["kind"] == "decision"]
    rng.shuffle(states)
    return states[:k]


def random_position(rng: random.Random) -> TableState:
    """A random valid mid-game position with assigned groups (covers unusual layouts)."""
    while True:
        n_mine, n_opp = rng.randint(1, 6), rng.randint(1, 6)
        mine_group = rng.choice(["solids", "stripes"])
        mine = rng.sample(SOLIDS if mine_group == "solids" else STRIPES, n_mine)
        opp = rng.sample(STRIPES if mine_group == "solids" else SOLIDS, n_opp)
        ids = mine + opp + ["8"]
        pts: dict[str, tuple[float, float]] = {}
        ok = True
        for bid in ids + ["cue"]:
            for _ in range(200):
                xy = (rng.uniform(1.5 * BALL_R, TABLE_W - 1.5 * BALL_R),
                      rng.uniform(1.5 * BALL_R, TABLE_L - 1.5 * BALL_R))
                if all(math.dist(xy, o) >= 2.2 * BALL_R for o in pts.values()):
                    pts[bid] = xy
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        cue = pts.pop("cue")
        opp_group = "stripes" if mine_group == "solids" else "solids"
        s = TableState(cue=cue, balls=pts, shooter="P1",
                       groups={"P1": mine_group, "P2": opp_group})
        if not validate(s):
            return s


# ----------------------------------------------------------------------------- tags


def difficulty(cands: list[Candidate]) -> str:
    best = max((c.rollouts.p_pot for c in cands if c.shot_type == "pot" and c.rollouts), default=0.0)
    if best >= 0.80:
        return "easy"
    if best >= 0.40:
        return "medium"
    return "hard"


def situation_tags(state: TableState, cands: list[Candidate]) -> list[str]:
    pots = [c for c in cands if c.shot_type == "pot" and c.rollouts]
    good = [c for c in pots if c.rollouts.p_pot >= 0.6]
    best = max((c.rollouts.p_pot for c in pots), default=0.0)
    tags = []
    if good:
        nexts = [c.rollouts.next_ease for c in good]
        if len(good) >= 2 and max(nexts) - min(nexts) >= 0.3:
            tags.append("position")
        else:
            tags.append("offense")
    from cuemind.sim.state import other

    if best < 0.4 or state.on_the_eight(other(state.shooter)):
        tags.append("defense")
    return tags or ["offense"]


# ----------------------------------------------------------------------------- build


def build_record(pid: str, split: str, source: str, state: TableState, prep: PrepConfig,
                 ocfg: OracleConfig) -> dict[str, Any] | None:
    state = resolve_ball_in_hand(state)
    cands = prepare(state, prep, rollout_key=("pos", pid))
    if len(cands) < 2:
        return None
    # Make sure every candidate also carries its L1 preview (prepare keeps it for L2).
    orc = oracle_values(state, cands, ocfg, key=("oracle", pid))
    return {
        "position_id": pid, "split": split, "source": source,
        "difficulty": difficulty(cands), "tags": situation_tags(state, cands),
        "state": state.model_dump(), "state_hash": state_hash(state),
        "prep": prep.to_dict(), "oracle_cfg": ocfg.__dict__,
        "candidates_hash": candidates_hash(cands),
        "candidates": [c.model_dump() for c in cands],
        "oracle": orc,
        "expected_intent": None, "acceptable_shot_ids": None, "reviewed_by": None,
    }


def load_positions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def menu_at_level(rec: dict, access_level: str) -> tuple[TableState, list[Candidate]]:
    """The frozen menu of a position, described at the requested access level."""
    state = TableState.model_validate(rec["state"])
    cands = [Candidate.model_validate(c) for c in rec["candidates"]]
    if access_level == "L0":
        cands = [c.model_copy(update={"preview": None, "rollouts": None}) for c in cands]
    elif access_level == "L1":
        cands = [c.model_copy(update={"rollouts": None}) for c in cands]
    return state, describe_all(state, cands, access_level)


# ----------------------------------------------------------------------------- scoring


def score_choice(rec: dict, shot_id: str) -> dict[str, float]:
    orc = rec["oracle"]
    vals = {k: v["eiv"] for k, v in orc.items()}
    best_id = max(vals, key=lambda k: (vals[k], k))
    best, chosen = vals[best_id], vals[shot_id]
    mean_all = sum(vals.values()) / len(vals)  # value of a uniformly random choice
    span = best - mean_all
    return {
        "eiv": chosen, "best_eiv": best, "regret": best - chosen,
        "top1": float(chosen >= best - orc[best_id]["se"]),
        "normalised": (chosen - mean_all) / span if span > 1e-9 else 1.0,
        "is_best": float(shot_id == best_id),
    }
