import json

from cuemind.candidates.pipeline import PrepConfig
from cuemind.eval.metrics import game_rows, pairing_summary
from cuemind.eval.tournament import build_jobs, run_tournament


def test_paired_tournament_runs_resumes_and_summarises(tmp_path):
    jobs = build_jobs("t", [("b1", "b0")], seeds=[1, 2], noise="medium",
                      prep=PrepConfig(access_level="L0"), max_shots=6)
    assert len(jobs) == 4 and len({j.game_id for j in jobs}) == 4
    res = run_tournament(jobs, tmp_path, workers=2, progress=False)
    assert len(res) == 4
    lines = (tmp_path / "games.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all((tmp_path / "raw" / f"{j.game_id}.jsonl").exists() for j in jobs)

    # Resume: nothing new is run.
    again = run_tournament(jobs, tmp_path, workers=2, progress=False)
    assert len(again) == 4
    assert len((tmp_path / "games.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    df = game_rows(res)
    assert set(df.system) == {"b1", "b0"} and len(df) == 8
    s = pairing_summary(res, "b1", "b0")
    assert s["games"] == 4 and s["paired_seeds"] == 2 and 0 <= s["win_rate"] <= 1
    g = json.loads(lines[0])
    assert "seat_metrics" in g and g["prep"]["access_level"] == "L0"


def test_midgame_tournament_swaps_the_shooter_seat(tmp_path):
    from cuemind.eval.tournament import build_midgame_jobs
    from cuemind.sim.state import TableState

    states = [TableState(cue=(0.5, 0.6), balls={"1": (0.2, 1.7), "9": (0.75, 1.2), "8": (0.5, 1.4)},
                         groups={"P1": "solids", "P2": "stripes"}, shooter=s).model_dump()
              for s in ("P1", "P2")]
    recs = [{"position_id": f"p{i}", "state": st} for i, st in enumerate(states)]
    jobs = build_midgame_jobs("m", [("b2", "b1")], recs, "medium", PrepConfig(access_level="L0"),
                              max_shots=4)
    assert len(jobs) == 4 and len({j.game_id for j in jobs}) == 4
    # whoever is listed first takes the shooter's seat
    j = jobs[2]  # second position (P2 to shoot), b2 first
    assert j.start_id.endswith("b2first") and j.p2 == "b2"
    res = run_tournament(jobs, tmp_path, workers=2, progress=False)
    assert len(res) == 4 and all(g["n_shots"] <= 4 for g in res)
