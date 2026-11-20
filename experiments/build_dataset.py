"""Build the positions benchmark (plan section 9).

    python experiments/build_dataset.py --config experiments/configs/dataset.yaml
    python experiments/build_dataset.py --config experiments/configs/dataset.yaml --dev 4 --test 6   # quick check

Steps: play B2-vs-B1 games (geometry-level menus, medium noise) to sample realistic
mid-game states; add random valid placements; then for every position build the frozen
L2-annotated menu, difficulty and situation tags, and oracle values (parallel). Positions
are stratified by difficulty for the test split. Output: data/positions/{dev,test}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from cuemind.candidates.pipeline import PrepConfig
from cuemind.eval.oracle import OracleConfig
from cuemind.eval.positions import build_record, random_position, sample_from_game
from cuemind.sim.state import TableState, state_hash

ROOT = Path(__file__).resolve().parents[1]


def _init():
    from cuemind.sim.execute import warm_up

    warm_up()


def _sample_game(seed: int, noise: str, per_game: int) -> list[dict]:
    from cuemind.game.runner import play_game
    from cuemind.policies.registry import make_policy

    res = play_game({"P1": make_policy("b2"), "P2": make_policy("b1")}, seed=seed, noise=noise,
                    prep=PrepConfig(access_level="L1"), max_shots=80)
    rng = random.Random(f"sample-{seed}")
    return [s.model_dump() for s in sample_from_game(res.records, rng, per_game)]


def _build(args):
    pid, split, source, state, prep, ocfg = args
    return build_record(pid, split, source, TableState.model_validate(state), PrepConfig(**prep),
                        OracleConfig(**ocfg))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dev", type=int, default=None)
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    n_dev = a.dev if a.dev is not None else cfg["n_dev"]
    n_test = a.test if a.test is not None else cfg["n_test"]
    workers = a.workers or cfg.get("workers", 8)
    out = Path(a.out) if a.out else ROOT / cfg.get("out_dir", "data/positions")
    out.mkdir(parents=True, exist_ok=True)
    prep = PrepConfig(**cfg.get("prep", {})).to_dict()
    ocfg = OracleConfig(**cfg.get("oracle", {})).__dict__
    rng = random.Random(cfg.get("seed", 0))

    # 1. Candidate states: sampled from games + random placements (over-generate x2).
    want = int((n_dev + n_test) * 2.0) + 4
    n_random = int(want * cfg.get("random_fraction", 0.2))
    n_games = (want - n_random) // cfg.get("per_game", 2) + 1
    t0 = time.perf_counter()
    pool_states: list[tuple[str, dict]] = []
    seeds = list(range(cfg["game_seed_start"], cfg["game_seed_start"] + n_games))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = [pool.submit(_sample_game, s, cfg.get("noise", "medium"), cfg.get("per_game", 2))
                for s in seeds]
        for f in as_completed(futs):
            pool_states += [("midgame_sample", s) for s in f.result()]
    pool_states += [("random", random_position(rng).model_dump()) for _ in range(n_random)]
    seen, uniq = set(), []
    for src, s in pool_states:
        h = state_hash(TableState.model_validate(s))
        if h not in seen:
            seen.add(h)
            uniq.append((src, s))
    uniq.sort(key=lambda t: state_hash(TableState.model_validate(t[1])))  # order-independent
    rng.shuffle(uniq)
    print(f"sampled {len(uniq)} unique states from {len(seeds)} games + {n_random} random "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)

    # 2. Annotate + oracle in parallel.
    jobs = [(f"pos_{i:04d}", "", src, s, prep, ocfg) for i, (src, s) in enumerate(uniq)]
    records = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = {pool.submit(_build, j): j[0] for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {futs[f]} failed: {exc!r}", flush=True)
                continue
            if r is not None:
                records.append(r)
            if k % 20 == 0:
                print(f"  annotated {k}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    records.sort(key=lambda r: r["position_id"])

    # 3. Split: dev first (any difficulty), then a difficulty-stratified test set.
    dev = records[:n_dev]
    rest = records[n_dev:]
    per = {d: [r for r in rest if r["difficulty"] == d] for d in ("easy", "medium", "hard")}
    test: list[dict] = []
    quota = n_test // 3
    for d in ("easy", "medium", "hard"):
        test += per[d][:quota]
    leftovers = [r for r in rest if r not in test]
    test += leftovers[: n_test - len(test)]
    for r in dev:
        r["split"] = "dev"
    for r in test:
        r["split"] = "test"
    for name, rows in (("dev", dev), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    stats = {name: {"n": len(rows), "difficulty": dict(Counter(r["difficulty"] for r in rows)),
                    "source": dict(Counter(r["source"] for r in rows)),
                    "tags": dict(Counter(t for r in rows for t in r["tags"])),
                    "mean_candidates": sum(len(r["candidates"]) for r in rows) / max(len(rows), 1)}
             for name, rows in (("dev", dev), ("test", test))}
    stats["config"] = cfg
    stats["build_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
