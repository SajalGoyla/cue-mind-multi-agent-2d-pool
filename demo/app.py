"""CueMind demo (v2): a single LLM agent and a team of LLM agents choose eight-ball shots.

Launch:  streamlit run demo/app.py

Modes (sidebar):
* Offline (fake LLM)   - no model at all; deterministic stand-in (always works).
* Local model (Ollama) - open-weight model on this machine (default qwen2.5:3b-instruct);
                         decisions made before return instantly and are labelled "cached".
Tabs: Decide a shot · Replay a game · Results.
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

from cuemind.candidates.model import Candidate  # noqa: E402
from cuemind.candidates.pipeline import PrepConfig, prepare  # noqa: E402
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


# ----------------------------------------------------------------------------- cached work
@st.cache_resource
def _warm() -> bool:
    warm_up()
    return True


@st.cache_data(show_spinner=False)
def menu_for(state_json: str) -> list[dict]:
    state = TableState.model_validate_json(state_json)
    return [c.model_dump() for c in prepare(state, PrepConfig(), rollout_key=("demo",))]


@st.cache_data(show_spinner=False)
def traj_for(state_json: str, stroke: tuple) -> dict:
    return trajectories(TableState.model_validate_json(state_json), Stroke(*stroke))


def get_policies(mode: str):
    from cuemind.policies.multi_agent import MultiAgent
    from cuemind.policies.single_agent import SingleAgent

    client = make_llm_client(llm="fake" if mode == "Offline (fake LLM)" else "ollama")
    return SingleAgent(client), MultiAgent(client)


def load_demo_positions() -> dict[str, dict]:
    data = json.loads((ROOT / "demo" / "sample_inputs.json").read_text(encoding="utf-8"))
    return {p["name"]: p["state"] for p in data["positions"]}


# ----------------------------------------------------------------------------- UI pieces
def usage_line(u: dict) -> str:
    cached = f", {int(u.get('cached_calls', 0))} cached" if u.get("cached_calls") else ""
    return (f"{int(u.get('calls', 0))} model calls{cached} · {int(u.get('input_tokens', 0)):,} in / "
            f"{int(u.get('output_tokens', 0)):,} out tokens · {u.get('latency_s', 0):.1f} s")


def decision_card(title: str, d, menu: list[Candidate]) -> None:
    st.markdown(f"#### {title}")
    if d is None:
        st.info("Not run yet.")
        return
    c = next((x for x in menu if x.shot_id == d.shot_id), None)
    st.markdown(f"**Choice: {d.shot_id}**" + (" (fallback)" if d.fallback else ""))
    if c is not None:
        st.caption(c.description)
    st.write(d.rationale)
    if d.usage:
        st.caption(usage_line(d.usage))


def shortlist_cards(d) -> None:
    shortlists = (d.trace or {}).get("shortlists") or {}
    if not shortlists:
        return
    cols = st.columns(len(shortlists))
    for col, (name, picks) in zip(cols, shortlists.items()):
        with col:
            st.markdown(f"**{name.title()} specialist**")
            if not picks:
                st.warning("no valid nominations")
            for i, p in enumerate(picks, 1):
                st.markdown(f"{i}. **{p['shot_id']}**: {p['reason']}")
    how = ("all three specialists ranked the same shot first (coordinator skipped)"
           if d.trace.get("unanimous") else
           f"coordinator chose; its choice was nominated by: {', '.join(d.trace.get('followed') or []) or 'none'}")
    st.caption(how)


def execute_panel(state: TableState, menu: list[Candidate], shot_id: str, noise: str, key: str) -> None:
    c = next(x for x in menu if x.shot_id == shot_id)
    executed = get_noise(noise).apply(c.stroke, rng_for("demo-exec", key, shot_id))
    traj = traj_for(state.model_dump_json(), (executed.phi, executed.V0, executed.a, executed.b, executed.theta))
    shot = simulate(to_system(state), executed)
    call = Call(c.target_ball, c.pocket) if c.shot_type == "pot" else Call(c.target_ball, None, True)
    j = judge(state, extract_events(shot), call)
    outcome = ("FOUL: " + j.foul) if j.foul else (
        f"potted {', '.join(j.potted)}" if j.potted else "nothing potted")
    st.plotly_chart(animation_figure(traj, title=f"{shot_id} played with {noise} noise: {outcome}"),
                    use_container_width=False)
    st.caption(f"Turn {'continues' if not j.turn_over else 'passes'}.")


# ----------------------------------------------------------------------------- tabs
def tab_decide(mode: str, noise: str) -> None:
    positions = load_demo_positions()
    name = st.selectbox("Position", list(positions))
    state = TableState.model_validate(positions[name])
    problems = validate(state)
    if problems:
        st.error("Invalid position: " + "; ".join(problems))
        return
    with st.spinner("Generating and forecasting every speed/spin variant..."):
        menu = [Candidate.model_validate(c) for c in menu_for(state.model_dump_json())]
    key = f"{name}|{mode}"
    res = st.session_state.get("decisions", {}).get(key)

    left, right = st.columns([1, 1.25])
    with left:
        hi = res[1].shot_id if res and res[1] is not None else None
        st.plotly_chart(table_figure(state, menu, highlight=hi, title=name), use_container_width=False)
    with right:
        st.markdown("**Shot menu** (what the agents see: geometry only, in shuffled order)")
        st.dataframe(pd.DataFrame([{"id": c.shot_id, "shot": c.description} for c in menu]),
                     hide_index=True, use_container_width=True, height=330)
        if st.button("Run single agent and multi-agent", type="primary"):
            sa, ma = get_policies(mode)
            ctx = DecisionContext(state=state, candidates=menu)
            out = []
            for label, pol in (("Single agent", sa), ("Multi-agent", ma)):
                with st.spinner(f"{label} is deciding..."):
                    try:
                        out.append(pol.decide(ctx))
                    except Exception as exc:  # noqa: BLE001 - show, never crash the demo
                        st.error(f"{label} failed: {type(exc).__name__}: {exc}")
                        out.append(None)
            out.append(SearchBot().decide(ctx))
            st.session_state.setdefault("decisions", {})[key] = tuple(out)
            st.rerun()

    if res:
        a, b, c = st.columns([1, 1.6, 1])
        with a:
            decision_card("Single agent", res[0], menu)
        with b:
            decision_card("Multi-agent team", res[1], menu)
            if res[1] is not None:
                shortlist_cards(res[1])
        with c:
            decision_card("Stat (reference, uses simulation)", res[2], menu)
        choices = sorted({d.shot_id for d in res if d is not None})
        pick = st.radio("Play which choice?", choices, horizontal=True)
        if st.button("Play shot"):
            execute_panel(state, menu, pick, noise, key)


def _game_files() -> list[str]:
    files = sorted(glob.glob(str(ROOT / "demo" / "games" / "*.jsonl")))
    return files + sorted(glob.glob(str(ROOT / "results" / "v2_*" / "raw" / "*.jsonl")))


def tab_replay() -> None:
    files = _game_files()
    if not files:
        st.info("No logged games found (demo/games/*.jsonl or results/v2_*/raw/*.jsonl).")
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
            st.markdown("**Break shot** (shared and seeded, not chosen by any player)")
        else:
            d = r["decision"]
            st.markdown(f"**{r['system']} chose {d['shot_id']}**" + (" (fallback)" if r.get("fallback") else ""))
            ch = next((c for c in cands if c.shot_id == d["shot_id"]), None)
            if ch:
                st.caption(ch.description)
            st.write(d.get("rationale", ""))
            for name, picks in ((r.get("trace") or {}).get("shortlists") or {}).items():
                st.markdown(f"- *{name}*: " + ", ".join(p["shot_id"] for p in picks))
        res = ("FOUL: " + o["foul"]) if o.get("foul") else (
            "potted " + ", ".join(o["potted"]) if o.get("potted") else "nothing potted")
        st.markdown(f"**Result:** {res}" + (f" · game over, winner {o['winner']}" if o.get("game_over") else ""))
        if st.button("Animate this shot"):
            ex = r["executed"]
            traj = traj_for(state.model_dump_json(), (ex["phi"], ex["V0"], ex["a"], ex["b"], ex["theta"]))
            st.plotly_chart(animation_figure(traj, title="Executed stroke"), use_container_width=False)


def tab_results() -> None:
    found = False
    md = ROOT / "results" / "summary.md"
    if md.exists():
        found = True
        st.markdown(md.read_text(encoding="utf-8"))
    for p in sorted(glob.glob(str(ROOT / "results" / "figures" / "*.png"))):
        found = True
        st.image(p, caption=Path(p).stem)
    if not found:
        st.info("No results yet. Run the experiments (see README).")


def main() -> None:
    _warm()
    st.title("CueMind: does a team of LLM agents play better pool than one LLM?")
    st.caption("Code generates the shot menu; the LLM systems only choose. They never see simulated forecasts.")
    with st.sidebar:
        mode = st.radio("LLM mode", ["Offline (fake LLM)", "Local model (Ollama)"])
        noise = st.selectbox("Stroke noise when playing a shot", ["low", "none"])
        if mode == "Local model (Ollama)":
            st.caption("Requires `ollama serve` and the model pulled; responses are cached.")
    t1, t2, t3 = st.tabs(["Decide a shot", "Replay a game", "Results"])
    with t1:
        tab_decide(mode, noise)
    with t2:
        tab_replay()
    with t3:
        tab_results()


main()
