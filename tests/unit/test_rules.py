from cuemind.game.rules import FOOT_SPOT, Call, advance, judge
from cuemind.sim.execute import ShotEvents
from cuemind.sim.state import TableState

BALLS = {"1": (0.2, 0.2), "2": (0.3, 0.9), "9": (0.4, 0.4), "8": (0.6, 0.6)}
OPEN = TableState(cue=(0.8, 1.5), balls=BALLS)
SOLIDS = TableState(cue=(0.8, 1.5), balls=BALLS, groups={"P1": "solids", "P2": "stripes"})
ON_8 = TableState(cue=(0.8, 1.5), balls={"9": (0.4, 0.4), "8": (0.6, 0.6)},
                  groups={"P1": "solids", "P2": "stripes"})
BREAK_CALL = Call(None, None, True)


def ev(first="1", pocketed=None, cushion=True, **kw):
    return ShotEvents(first_ball_hit=first, pocketed=pocketed or {}, cushion_after_contact=cushion,
                      **kw)


def test_called_pot_continues_turn():
    j = judge(SOLIDS, ev(pocketed={"1": "lb"}), Call("1", "lb"))
    assert j.legal and not j.turn_over and j.called_ball_made


def test_wrong_pocket_ends_turn_but_is_legal():
    j = judge(SOLIDS, ev(pocketed={"1": "rb"}), Call("1", "lb"))
    assert j.legal and j.turn_over and not j.called_ball_made


def test_fouls():
    assert "scratch" in judge(SOLIDS, ev(pocketed={"cue": "lc"}), Call("1", "lb")).foul
    assert judge(SOLIDS, ev(first=None, cushion=False), Call("1", "lb")).foul == "no ball contacted"
    assert "wrong first contact" in judge(SOLIDS, ev(first="9"), Call("1", "lb")).foul
    assert judge(SOLIDS, ev(cushion=False), Call("1", "lb")).foul == "no cushion after contact"
    # hitting the 8 first on an open table is a foul
    assert "wrong first contact" in judge(OPEN, ev(first="8"), Call("1", "lb")).foul


def test_safety_ends_turn_even_if_ball_drops():
    j = judge(SOLIDS, ev(pocketed={"1": "lb"}), Call("1", None, safety=True))
    assert j.legal and j.turn_over


def test_open_table_assigns_called_group():
    j = judge(OPEN, ev(first="9", pocketed={"9": "rt", "1": "lb"}), Call("9", "rt"))
    assert j.group_for_shooter == "stripes"
    nxt = advance(OPEN, j, (0.5, 0.5), {"2": (0.3, 0.9), "8": (0.6, 0.6)})
    assert nxt.groups == {"P1": "stripes", "P2": "solids"} and nxt.shooter == "P1"


def test_eight_ball_win_and_losses():
    win = judge(ON_8, ev(first="8", pocketed={"8": "rt"}), Call("8", "rt"))
    assert win.game_over and win.winner == "P1"
    assert judge(ON_8, ev(first="8", pocketed={"8": "lt"}), Call("8", "rt")).winner == "P2"
    scratch = judge(ON_8, ev(first="8", pocketed={"8": "rt", "cue": "lb"}), Call("8", "rt"))
    assert scratch.winner == "P2"
    early = judge(SOLIDS, ev(first="1", pocketed={"8": "rt"}), Call("1", "lb"))
    assert early.game_over and early.winner == "P2"


def test_foul_gives_ball_in_hand_to_opponent():
    j = judge(SOLIDS, ev(first="9"), Call("1", "lb"))
    nxt = advance(SOLIDS, j, (0.5, 0.5), BALLS)
    assert nxt.shooter == "P2" and nxt.ball_in_hand and nxt.cue is None


def test_break_rules():
    brk = OPEN.model_copy(update={"break_shot": True})
    assert "scratch" in judge(brk, ev(pocketed={"cue": "lb", "3": "rt"}), BREAK_CALL).foul
    assert "illegal break" in judge(brk, ev(object_balls_hit_cushion=2), BREAK_CALL).foul
    ok = judge(brk, ev(object_balls_hit_cushion=5), BREAK_CALL)
    assert ok.legal and ok.turn_over  # nothing dropped: turn passes
    made = judge(brk, ev(pocketed={"3": "rt"}), BREAK_CALL)
    assert made.legal and not made.turn_over
    eight = judge(brk, ev(pocketed={"8": "rt"}), BREAK_CALL)
    assert not eight.game_over
    nxt = advance(brk, eight, (0.5, 0.5), {"1": (0.2, 0.2)})
    assert "8" in nxt.balls and abs(nxt.balls["8"][0] - FOOT_SPOT[0]) < 1e-9
    assert nxt.groups == {"P1": "open", "P2": "open"}
