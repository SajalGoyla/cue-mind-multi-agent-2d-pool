import json

from cuemind.candidates.pipeline import MenuConfig
from cuemind.eval.metrics import game_rows, pairing_summary
from cuemind.eval.tournament import build_jobs, run_tournament


def test_paired_tournament_runs_resumes_and_summarises(tmp_path):
    jobs = build_jobs("t", [("b1", "b"), ("b1", "b1")], seeds=[1, 2], noise="none",
                      menu=MenuConfig(), max_shots=6)
    # two seat orders per seed for b1 vs b; one game per seed for self-play
    assert len(jobs) == 6 and len({j.game_id for j in jobs}) == 6
    res = run_tournament(jobs, tmp_path, workers=2, progress=False)
    assert len(res) == 6
    lines = (tmp_path / "games.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert all((tmp_path / "raw" / f"{j.game_id}.jsonl").exists() for j in jobs)

    again = run_tournament(jobs, tmp_path, workers=2, progress=False)  # resume: nothing new
    assert len(again) == 6 and len((tmp_path / "games.jsonl").read_text(encoding="utf-8").splitlines()) == 6

    df = game_rows(res)
    assert set(df.system) == {"b1", "b"}
    s = pairing_summary(res, "b1", "b")
    assert s["games"] == 4 and s["paired_seeds"] == 2 and 0 <= s["win_rate"] <= 1
    assert pairing_summary(res, "b1", "b1")["games"] == 2
    g = json.loads(lines[0])
    assert "seat_metrics" in g and g["menu"]["max_pot_targets"] == 4 and "ending" in g
