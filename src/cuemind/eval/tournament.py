"""Paired, seeded, resumable, parallel tournaments between policies.

For every seed and every pairing (X, Y) two games are played: X breaks (seat P1) and Y
breaks. Both games use the same rack, break jitter and execution-noise streams (common random
numbers, D-004), so the seed pair compares decisions, not luck. Because every pairing uses the
same seeds, two systems that both play the reference B can also be compared seed by seed.

Outputs (per experiment directory):
    games.jsonl          one summary line per finished game (append-only; used for resume)
    raw/<game_id>.jsonl  every shot record of that game (git-ignored)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cuemind.candidates.pipeline import MenuConfig
from cuemind.eval.metrics import seat_metrics


@dataclass
class GameJob:
    exp_id: str
    seed: int
    p1: str
    p2: str
    noise: str
    menu: dict
    max_shots: int = 150
    policy_kwargs: dict = field(default_factory=dict)

    @property
    def game_id(self) -> str:
        return f"{self.exp_id}_{self.p1}_vs_{self.p2}_{self.noise}_s{self.seed}"


def build_jobs(exp_id: str, pairings: list[tuple[str, str]], seeds: list[int], noise: str,
               menu: MenuConfig, max_shots: int = 150,
               policy_kwargs: dict | None = None) -> list[GameJob]:
    jobs = []
    for a, b in pairings:
        # Self-play (B vs B) needs one game per seed: swapping identical seats replays it exactly.
        orders = ((a, b),) if a == b else ((a, b), (b, a))
        for s in seeds:
            for p1, p2 in orders:
                jobs.append(GameJob(exp_id, s, p1, p2, noise, menu.to_dict(), max_shots,
                                    policy_kwargs or {}))
    return jobs


def run_job(job: GameJob, raw_dir: str | None) -> dict[str, Any]:
    """Executed in a worker process."""
    from cuemind.game.runner import play_game
    from cuemind.policies.registry import make_policy

    kw = job.policy_kwargs
    policies = {"P1": make_policy(job.p1, **kw.get(job.p1, {})),
                "P2": make_policy(job.p2, **kw.get(job.p2, {}))}
    log_path = None
    if raw_dir:
        log_path = Path(raw_dir) / f"{job.game_id}.jsonl"
        log_path.unlink(missing_ok=True)  # restart cleanly if a previous attempt died
    res = play_game(policies, seed=job.seed, noise=job.noise, menu=MenuConfig.from_dict(job.menu),
                    max_shots=job.max_shots, game_id=job.game_id, log_path=log_path,
                    exp_id=job.exp_id)
    summary = res.summary()
    summary.update(players={"P1": job.p1, "P2": job.p2}, breaker="P1", noise=job.noise,
                   menu=job.menu, seat_metrics=seat_metrics(res.records),
                   ending=_ending(res.records))
    return summary


def _ending(records: list[dict]) -> str | None:
    """How the game ended: 'legal 8', '8 with foul', '8 early or wrong pocket', or None
    if it did not end (shot cap or error)."""
    if not records or not records[-1]["outcome"]["game_over"]:
        return None
    o = records[-1]["outcome"]
    if o["legal"]:
        return "legal 8"
    return "8 with foul" if o["foul"] else "8 early or wrong pocket"


def _init_worker() -> None:
    from cuemind.sim.execute import warm_up

    warm_up()


def load_done(games_path: Path) -> dict[str, dict]:
    done = {}
    if games_path.exists():
        for line in games_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                if g.get("status") != "error":
                    done[g["game_id"]] = g
    return done


def run_tournament(jobs: list[GameJob], out_dir: Path, workers: int = 8,
                   keep_raw: bool = True, progress: bool = True) -> list[dict]:
    """Run all jobs not already finished in out_dir/games.jsonl. Returns all summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "games.jsonl"
    raw_dir = out_dir / "raw"
    if keep_raw:
        raw_dir.mkdir(exist_ok=True)
    done = load_done(games_path)
    todo = [j for j in jobs if j.game_id not in done]
    if progress:
        print(f"{len(jobs)} games in total, {len(done)} already done, {len(todo)} to run "
              f"with {workers} workers", flush=True)
    t0 = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futs = {pool.submit(run_job, j, str(raw_dir) if keep_raw else None): j for j in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                job = futs[fut]
                try:
                    summary = fut.result()
                except Exception as exc:  # noqa: BLE001 - a crashed worker must not stop the run
                    summary = {"game_id": job.game_id, "seed": job.seed, "status": "error",
                               "error": f"worker crash: {exc!r}",
                               "players": {"P1": job.p1, "P2": job.p2}, "winner": None,
                               "n_shots": 0}
                with open(games_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, default=str) + "\n")
                if summary.get("status") != "error":
                    done[job.game_id] = summary
                if progress:
                    el = time.perf_counter() - t0
                    print(f"[{k}/{len(todo)}] {job.game_id}: {summary.get('status')} "
                          f"winner={summary.get('winner')} shots={summary.get('n_shots')} "
                          f"({el / k * (len(todo) - k) / 60:.1f} min left)", flush=True)
    return [done[j.game_id] for j in jobs if j.game_id in done]
