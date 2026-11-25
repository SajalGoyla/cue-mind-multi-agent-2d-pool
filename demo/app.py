"""CueMind demo: B, a single agent and a team of agents choose a pool shot from the same raw menu.

Launch (from the repository root):  streamlit run demo/app.py

Modes (sidebar):
* Offline (fake LLM)   - no model at all; deterministic stand-in (always works).
* Local model (Ollama) - open-weight model on this machine (default qwen2.5:7b-instruct);
                         decisions made before come back instantly from the response cache.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from cuemind.candidates.model import Candidate  # noqa: E402
from cuemind.candidates.pipeline import build_menu  # noqa: E402
from cuemind.game.rules import Call, judge  # noqa: E402
from cuemind.policies.base import DecisionContext  # noqa: E402
from cuemind.policies.registry import make_llm_client  # noqa: E402
from cuemind.policies.search_bot import SearchBot  # noqa: E402
from cuemind.sim.execute import Stroke, extract_events, simulate, warm_up  # noqa: E402
from cuemind.sim.noise import get_noise, rng_for  # noqa: E402
from cuemind.sim.state import TableState, to_system, validate  # noqa: E402
from cuemind.sim.trajectories import trajectories  # noqa: E402
from cuemind.viz.table_plot import animation_figure, table_figure  # noqa: E402

st.set_page_config(page_title="CueMind", layout="wide")
MODES = ["Offline (fake LLM)", "Local model (Ollama)"]


# ----------------------------------------------------------------------------- cached work
@st.cache_resource
def _warm() -> bool:
    warm_up()
    return True


@st.cache_data(show_spinner=False)
def menu_for(state_json: str, key: str) -> list[dict]:
    state = TableState.model_validate_json(state_json)
    return [c.model_dump() for c in build_menu(state, menu_key=("demo", key))]


@st.cache_data(show_spinner=False)
def traj_for(state_json: str, stroke: tuple) -> dict:
    return trajectories(TableState.model_validate_json(state_json), Stroke(*stroke))


def get_policies(mode: str):
    from cuemind.policies.multi_agent import MultiAgent
    from cuemind.policies.single_agent import SingleAgent

    client = make_llm_client(llm="fake" if mode == MODES[0] else "ollama")
    return SearchBot(), SingleAgent(client), MultiAgent(client)


def load_demo_positions() -> dict[str, dict]:
    data = json.loads((ROOT / "demo" / "sample_inputs.json").read_text(encoding="utf-8"))
    return {p["name"]: p["state"] for p in data["positions"]}


# ----------------------------------------------------------------------------- UI pieces
def usage_line(u: dict) -> str:
    if not u.get("calls"):
        return "no LLM calls"
    cached = f", {int(u.get('cached_calls', 0))} cached" if u.get("cached_calls") else ""
    return (f"{int(u.get('calls', 0))} LLM calls{cached} · {int(u.get('input_tokens', 0)):,} in / "
            f"{int(u.get('output_tokens', 0)):,} out tokens · {u.get('latency_s', 0):.1f} s")


def decision_card(title: str, d, menu: list[Candidate]) -> None:
    st.markdown(f"#### {title}")
    if d is None:
        st.info("Failed; see the message above.")
        return
    c = next((x for x in menu if x.shot_id == d.shot_id), None)
    badge = " (fallback)" if d.fallback else ""
    st.markdown(f"**Choice: {d.shot_id}{badge}**, intent *{d.intent}*")
    if c is not None:
        st.caption(c.description)
    st.write(d.rationale)
    st.caption(usage_line(d.usage))


def b_details(d) -> None:
    sim = (d.trace or {}).get("sim", {}).get(d.shot_id)
    if sim:
        st.caption(f"B's private simulation of {d.shot_id} ({d.trace['n_rollouts']} run(s)): pots "
                   f"{sim['p_pot']:.0%}, scratch {sim['p_scratch']:.0%}, foul {sim['p_foul']:.0%}. "
                   "SA and MA never see these numbers.")


def specialist_cards(d) -> None:
    reports = (d.trace or {}).get("reports") or {}
    if not reports:
        return
    cols = st.columns(len(reports))
    for col, (name, r) in zip(cols, reports.items()):
        with col:
            st.markdown(f"**{name.title()} specialist**")
            if "error" in r:
                st.warning(r["error"])
                continue
            for rank, item in enumerate(r["shortlist"], 1):
                st.markdown(f"{rank}. **{item['shot_id']}** — {item['reason']}")
    if d.trace.get("unanimous"):
        st.caption("All three ranked the same shot first, so no coordinator call was needed.")
    else:
        st.caption(f"Coordinator chose from {', '.join(d.trace.get('shortlisted', []))}; "
                   f"followed: {', '.join(d.trace.get('followed') or []) or 'not stated'}.")


def execute_panel(state: TableState, menu: list[Candidate], shot_id: str, noise: str, key: str) -> None:
    c = next(x for x in menu if x.shot_id == shot_id)
    executed = get_noise(noise).apply(c.stroke, rng_for("demo-exec", key, shot_id))
    sj = state.model_dump_json()
    traj = traj_for(sj, (executed.phi, executed.V0, executed.a, executed.b, executed.theta))
    shot = simulate(to_system(state), executed)
    call = Call(c.target_ball, c.pocket) if c.shot_type == "pot" else Call(c.target_ball, None, True)
    j = judge(state, extract_events(shot), call)
    outcome = ("FOUL: " + j.foul) if j.foul else (
        f"potted {', '.join(j.potted)}" if j.potted else "nothing potted")
    st.plotly_chart(animation_figure(traj, title=f"{shot_id} executed with {noise} noise: {outcome}"),
                    use_container_width=False)
    st.caption(f"Aim error {executed.phi - c.phi:+.2f}°, speed {executed.V0:.2f} m/s "
               f"(planned {c.V0:.2f}). Turn {'continues' if not j.turn_over else 'passes'}.")


# ----------------------------------------------------------------------------- tabs
def tab_decide(mode: str, noise: str) -> None:
    positions = load_demo_positions()
    name = st.selectbox("Position", list(positions))
    state = TableState.model_validate(positions[name])
    problems = validate(state)
    if problems:
        st.error("Invalid position: " + "; ".join(problems))
        return
    with st.spinner("Generating candidate shots..."):
        menu = [Candidate.model_validate(c) for c in menu_for(state.model_dump_json(), name)]
    key = f"{name}|{mode}|{noise}"
    res = st.session_state.get("decisions", {}).get(key)

    left, right = st.columns([1, 1.25])
    with left:
        hi = None
        if res:
            hi = next((d.shot_id for d in (res[2], res[1], res[0]) if d is not None), None)
        st.plotly_chart(table_figure(state, menu, highlight=hi, title=name), use_container_width=False)
    with right:
        st.markdown("**Candidate shots** (geometry only; the list order is random and means nothing)")
        st.dataframe(pd.DataFrame([{"id": c.shot_id, "shot": c.description}
                                   for c in sorted(menu, key=lambda c: c.shot_id)]),
                     hide_index=True, use_container_width=True, height=330)
        if st.button("Run B, single agent and multi-agent", type="primary"):
            ctx = DecisionContext(state=state, candidates=menu, noise=noise, sim_key=("demo", name))
            out = []
            for label, pol in zip(("B", "Single agent", "Multi-agent"), get_policies(mode)):
                with st.spinner(f"{label} is deciding..."):
                    try:
                        out.append(pol.decide(ctx))
                    except Exception as exc:  # noqa: BLE001 - show, never crash the demo
                        st.error(f"{label} failed: {type(exc).__name__}: {exc}")
                        out.append(None)
            st.session_state.setdefault("decisions", {})[key] = tuple(out)
            st.rerun()

    if res:
        a, b, c = st.columns(3)
        with a:
            decision_card("B (simulates every shot)", res[0], menu)
            if res[0] is not None:
                b_details(res[0])
        with b:
            decision_card("Single agent", res[1], menu)
        with c:
            decision_card("Multi-agent", res[2], menu)
        if res[2] is not None:
            specialist_cards(res[2])
        choices = sorted({d.shot_id for d in res if d is not None})
        if choices:
            pick = st.radio("Execute which choice?", choices, horizontal=True)
            if st.button("Execute shot"):
                execute_panel(state, menu, pick, noise, key)


def _game_files() -> list[str]:
    files = sorted(glob.glob(str(ROOT / "demo" / "games" / "*.jsonl")))
    files += sorted(glob.glob(str(ROOT / "results" / "*" / "raw" / "*.jsonl")))
    return files


def tab_replay() -> None:
    files = _game_files()
    if not files:
        st.info("No logged games found (demo/games/*.jsonl or results/*/raw/*.jsonl).")
        return
    f = st.selectbox("Game log", files, format_func=lambda p: Path(p).name)
    recs = [json.loads(x) for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not recs:
        st.warning("Empty log.")
        return
    i = st.slider("Shot", 0, len(recs) - 1, 0)
    r = recs[i]
    state = TableState.model_validate(r["state"])
    cands = [Candidate.model_validate(c) for c in r.get("candidates", [])]
    o = r["outcome"]
    left, right = st.columns([1, 1.25])
    with left:
        hi = (r.get("decision") or {}).get("shot_id")
        st.plotly_chart(table_figure(state, cands, highlight=hi,
                                     title=f"Shot {r['shot_index']}: {r['seat']} ({r['system']})"),
                        use_container_width=False)
    with right:
        if r["kind"] == "break":
            st.markdown("**Break shot** (shared and seeded; not chosen by any player)")
        else:
            d = r["decision"]
            st.markdown(f"**{r['system']} chose {d['shot_id']}** ({d['intent']})"
                        + (" — fallback" if r.get("fallback") else ""))
            ch = next((c for c in cands if c.shot_id == d["shot_id"]), None)
            if ch:
                st.caption(ch.description)
            st.write(d.get("rationale", ""))
            for name, rep in ((r.get("trace") or {}).get("reports") or {}).items():
                picks = ", ".join(f"{x['shot_id']} ({x['reason']})" for x in rep.get("shortlist", []))
                st.markdown(f"- *{name}*: {picks or rep.get('error', '')}")
        res = ("FOUL: " + o["foul"]) if o.get("foul") else (
            "potted " + ", ".join(o["potted"]) if o.get("potted") else "nothing potted")
        st.markdown(f"**Result:** {res}" + (" · game over, winner " + str(o['winner']) if o.get("game_over") else ""))
        if st.button("Animate this shot"):
            ex = r["executed"]
            traj = traj_for(state.model_dump_json(), (ex["phi"], ex["V0"], ex["a"], ex["b"], ex["theta"]))
            st.plotly_chart(animation_figure(traj, title="Executed stroke"), use_container_width=False)


def tab_results() -> None:
    summary = ROOT / "results" / "summary.md"
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))
    for p in sorted(glob.glob(str(ROOT / "results" / "figures" / "*.png"))):
        st.image(p, caption=Path(p).stem)
    if not summary.exists():
        st.info("No results yet. Run the experiments, then `python experiments/analyze.py`.")


def main() -> None:
    _warm()
    st.title("CueMind: does a team of LLM agents play better pool than one LLM?")
    st.caption("Code lists the candidate shots from geometry; B simulates them privately; the LLM "
               "systems judge from the geometry alone.")
    with st.sidebar:
        mode = st.radio("LLM mode", MODES)
        noise = st.selectbox("Execution noise", ["none", "low", "medium"])
        if mode == MODES[1]:
            st.caption("Requires `ollama serve` and the model pulled; responses are cached.")
    t1, t2, t3 = st.tabs(["Decide a shot", "Replay a game", "Results"])
    with t1:
        tab_decide(mode, noise)
    with t2:
        tab_replay()
    with t3:
        tab_results()


main()
