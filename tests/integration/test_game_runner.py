import json

from cuemind.game.runner import play_game
from cuemind.policies.base import Decision
from cuemind.policies.registry import make_policy
from cuemind.policies.search_bot import SearchBot
from cuemind.policies.simple_bots import GeometricBot
from cuemind.sim.state import TableState


class Exploding:
    name = "exploding"

    def decide(self, ctx):
        raise RuntimeError("boom")


class BadId:
    name = "bad_id"

    def decide(self, ctx):
        return Decision(shot_id="S99")


def test_b_vs_b1_game_runs_and_logs(tmp_path):
    log = tmp_path / "g.jsonl"
    res = play_game({"P1": SearchBot(), "P2": GeometricBot()}, seed=0, noise="none",
                    max_shots=30, log_path=log)
    assert res.status in ("finished", "shot_cap"), res.error
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == res.n_shots and lines[0]["kind"] == "break"
    dec = [r for r in lines if r["kind"] == "decision"]
    assert dec
    assert all(r["decision"]["shot_id"] in {c["shot_id"] for c in r["candidates"]} for r in dec)
    assert all(c["rollouts"] is None for r in dec for c in r["candidates"])  # raw menus only
    assert any("ev" in r["trace"] for r in dec if r["system"] == "B_search")


def test_policy_failures_fall_back_without_crashing():
    res = play_game({"P1": Exploding(), "P2": BadId()}, seed=1, max_shots=12)
    assert res.status in ("finished", "shot_cap"), res.error
    dec = [r for r in res.records if r["kind"] == "decision"]
    assert dec and all(r["fallback"] for r in dec)


def test_mid_game_start_and_shot_cap():
    start = TableState(cue=(0.5, 0.6), balls={"1": (0.20, 1.70), "9": (0.75, 1.2), "8": (0.5, 1.4)},
                       groups={"P1": "solids", "P2": "stripes"}, shooter="P2")
    res = play_game({"P1": GeometricBot(), "P2": GeometricBot()}, seed=2, start_state=start,
                    max_shots=2)
    assert res.records[0]["seat"] == "P2" and res.records[0]["kind"] == "decision"
    assert res.n_shots <= 2


def test_repeated_games_are_reproducible():
    """Same seed and same choices -> identical executed strokes (common random numbers)."""
    a = play_game({"P1": GeometricBot(), "P2": GeometricBot()}, seed=5, noise="low", max_shots=6)
    b = play_game({"P1": GeometricBot(), "P2": GeometricBot()}, seed=5, noise="low", max_shots=6)
    assert [r["executed"] for r in a.records] == [r["executed"] for r in b.records]


def test_llm_policies_play_offline():
    res = play_game({"P1": make_policy("sa", llm="fake"), "P2": make_policy("ma", llm="fake")},
                    seed=4, max_shots=8)
    assert res.status in ("finished", "shot_cap"), res.error
    ma = [r for r in res.records if r["kind"] == "decision" and r["system"] == "MA"]
    assert all("reports" in r["trace"] and not r["fallback"] for r in ma)
