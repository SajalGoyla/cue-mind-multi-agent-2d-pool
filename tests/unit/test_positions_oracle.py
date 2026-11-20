import random

import pytest

from cuemind.candidates.model import Candidate
from cuemind.candidates.pipeline import PrepConfig
from cuemind.eval.oracle import OracleConfig, eiv
from cuemind.eval.positions import (build_record, menu_at_level, random_position, score_choice,
                                    situation_tags)
from cuemind.policies.base import DecisionContext
from cuemind.policies.search_bot import SearchBot
from cuemind.sim.state import TableState, validate

HANGER = TableState(cue=(0.40, 1.40), balls={"1": (0.12, 1.86), "2": (0.6, 0.5), "9": (0.8, 0.4),
                                              "8": (0.7, 1.0)}, groups={"P1": "solids", "P2": "stripes"})
PREP = PrepConfig(access_level="L2", n_rollouts=4)
ORC = OracleConfig(m=6, horizon=1)


@pytest.fixture(scope="module")
def rec():
    r = build_record("t0", "test", "hand", HANGER, PREP, ORC)
    assert r is not None
    return r


def test_record_has_frozen_menu_tags_and_oracle(rec):
    ids = [c["shot_id"] for c in rec["candidates"]]
    assert set(rec["oracle"]) == set(ids)
    assert rec["difficulty"] in ("easy", "medium", "hard") and rec["tags"]
    assert all(c["rollouts"] and c["preview"] for c in rec["candidates"])


def test_hanging_pot_is_worth_more_than_a_safety(rec):
    cands = [Candidate.model_validate(c) for c in rec["candidates"]]
    pot = max((c for c in cands if c.shot_type == "pot" and c.target_ball == "1"),
              key=lambda c: rec["oracle"][c.shot_id]["eiv"])
    safeties = [c for c in cands if c.shot_type == "safety"]
    if safeties:
        assert rec["oracle"][pot.shot_id]["eiv"] > max(rec["oracle"][s.shot_id]["eiv"] for s in safeties)
    assert rec["oracle"][pot.shot_id]["eiv"] > 0.5


def test_menu_at_level_hides_information(rec):
    _, l0 = menu_at_level(rec, "L0")
    _, l1 = menu_at_level(rec, "L1")
    _, l2 = menu_at_level(rec, "L2")
    assert [c.shot_id for c in l0] == [c.shot_id for c in l2]
    assert all(c.rollouts is None and c.preview is None for c in l0)
    assert all(c.rollouts is None and c.preview is not None for c in l1)
    assert "realistic strokes" in l2[0].description and "realistic strokes" not in l0[0].description
    assert "preview" in l1[0].description


def test_scoring(rec):
    best = max(rec["oracle"], key=lambda k: (rec["oracle"][k]["eiv"], k))
    s = score_choice(rec, best)
    assert s["regret"] == 0 and s["top1"] == 1.0 and s["is_best"] == 1.0
    worst = min(rec["oracle"], key=lambda k: rec["oracle"][k]["eiv"])
    assert score_choice(rec, worst)["regret"] >= 0
    state, cands = menu_at_level(rec, "L2")
    d = SearchBot().decide(DecisionContext(state=state, candidates=cands))
    assert score_choice(rec, d.shot_id)["regret"] >= 0


def test_oracle_is_deterministic():
    from cuemind.candidates.pipeline import prepare

    c = prepare(HANGER, PREP, rollout_key=(1,))[0]
    assert eiv(HANGER, c, ORC, ("k",)) == eiv(HANGER, c, ORC, ("k",))


def test_random_positions_are_valid():
    rng = random.Random(3)
    for _ in range(5):
        s = random_position(rng)
        assert validate(s) == [] and s.groups["P1"] != "open" and "8" in s.balls


def test_situation_tags_defense_when_nothing_pots(rec):
    cands = [Candidate.model_validate(c) for c in rec["candidates"]]
    low = [c.model_copy(update={"rollouts": c.rollouts.model_copy(update={"p_pot": 0.1})})
           for c in cands]
    assert "defense" in situation_tags(HANGER, low)
