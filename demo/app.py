"""CueMind demo: watch a single agent and a team of agents choose a pool shot.

Launch:  streamlit run demo/app.py

Modes (sidebar):
* Offline (fake LLM)  - no network or credentials; deterministic stand-in for the LLM.
* Claude (live)       - real calls through Vertex AI (.env), with the response cache:
                        a decision already made before returns instantly and is labelled "cached".
* Claude (cache only) - never calls the API; only replays cached responses.
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
from cuemind.candidates.pipeline import PrepConfig, prepare  # noqa: E402
from cuemind.eval.positions import load_positions, menu_at_level, score_choice  # noqa: E402
from cuemind.game.rules import Call, judge  # noqa: E402
from cuemind.policies.base import DecisionContext  # noqa: E402
from cuemind.policies.registry import make_llm_client  # noqa: E402
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
def menu_for(state_json: str, access: str, noise: str) -> list[dict]:
    state = TableState.model_validate_json(state_json)
    cands = prepare(state, PrepConfig(access_level=access, noise=noise), rollout_key=("demo",))
    return [c.model_dump() for c in cands]


@st.cache_data(show_spinner=False)
def traj_for(state_json: str, stroke: tuple) -> dict:
    return trajectories(TableState.model_validate_json(state_json), Stroke(*stroke))


def get_policies(mode: str):
    from cuemind.policies.multi_agent import MultiAgent
    from cuemind.policies.single_agent import SingleAgent

    if mode == "Offline (fake LLM)":
        client = make_llm_client(llm="fake")
    else:
        client = make_llm_client(run_id="demo", max_usd=2.0, offline=(mode == "Claude (cache only)"))
    return SingleAgent(client), MultiAgent(client)


def load_demo_positions() -> dict[str, dict]:
    data = json.loads((ROOT / "demo" / "sample_inputs.json").read_text(encoding="utf-8"))
    out = {p["name"]: {"state": p["state"]} for p in data["positions"]}
    records = {}
    for f in ("test.jsonl", "hard_cases.jsonl"):
        path = ROOT / "data" / "positions" / f
        if path.exists():
            records.update({r["position_id"]: r for r in load_positions(path)})
    runs = ROOT / "results" / "demo_assets" / "sample_runs.json"
    if runs.exists():  # sample runs picked from real experiment logs (scripts/export_demo_assets.py)
        for run in json.loads(runs.read_text(encoding="utf-8"))["runs"]:
            rec = records.get(run["position_id"])
            entry = {"state": run["state"]}
            if rec is not None:
                entry["record"] = rec
            out[f"{run['slot']} ({run['position_id']}, from {runs.parent.parent.name} logs)"] = entry
    hard = ROOT / "data" / "positions" / "hard_cases.jsonl"
    if hard.exists():
        for rec in load_positions(hard):
            out[f"Hard case {rec['position_id']} ({rec.get('label_status', 'draft')})"] = {
                "state": rec["state"], "record": rec}
    test = ROOT / "data" / "positions" / "test.jsonl"
    if test.exists():
        for rec in load_positions(test)[:40]:
            out[f"Benchmark {rec['position_id']} ({rec['difficulty']}, {', '.join(rec['tags'])})"] = {
                "state": rec["state"], "record": rec}
    return out


# ----------------------------------------------------------------------------- UI pieces
def usage_line(u: dict) -> str:
    cached = f", {int(u.get('cached_calls', 0))} cached" if u.get("cached_calls") else ""
    return (f"{int(u.get('calls', 0))} calls{cached} · {int(u.get('input_tokens', 0)):,} in / "
            f"{int(u.get('output_tokens', 0)):,} out tokens · ${u.get('cost_usd', 0):.4f} · "
            f"{u.get('latency_s', 0):.1f} s")


def decision_card(title: str, d, menu: list[Candidate]) -> None:
    st.markdown(f"#### {title}")
    if d is None:
        st.info("Not run yet.")
        return
    c = next((x for x in menu if x.shot_id == d.shot_id), None)
    badge = " (fallback)" if d.fallback else ""
    st.markdown(f"**Choice: {d.shot_id}{badge}**, intent *{d.intent}*" +
                (f", confidence {d.confidence:.2f}" if d.confidence is not None else ""))
    if c is not None:
        st.caption(c.description)
    st.write(d.rationale)
    st.caption(usage_line(d.usage))


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
            st.markdown(f"Top choice: **{r.get('top_choice')}**")
            st.caption(r.get("summary", ""))
            scores = r.get("scores") or {}
            if scores:
                st.dataframe(pd.DataFrame({"score": scores}).T, use_container_width=True)
    dec_followed = d.trace.get("followed") or []
    over = d.trace.get("overruled") or []
    st.caption(f"Coordinator followed: {', '.join(dec_followed) or 'none'} · "
               f"overruled: {', '.join(over) or 'none'}")


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
def tab_decide(mode: str, access: str, noise: str) -> None:
    positions = load_demo_positions()
    name = st.selectbox("Position", list(positions))
    entry = positions[name]
    state = TableState.model_validate(entry["state"])
    problems = validate(state)
    if problems:
        st.error("Invalid position: " + "; ".join(problems))
        return
    if "record" in entry:
        state, menu = menu_at_level(entry["record"], access)
    else:
        with st.spinner("Generating and simulating candidate shots..."):
            menu = [Candidate.model_validate(c) for c in menu_for(state.model_dump_json(), access, noise)]
    key = f"{name}|{access}|{mode}"
    res = st.session_state.get("decisions", {}).get(key)

    left, right = st.columns([1, 1.25])
    with left:
        hi = None
        if res:
            hi = res[1].shot_id if res[1] is not None else res[0].shot_id
        st.plotly_chart(table_figure(state, menu, highlight=hi, title=name), use_container_width=False)
    with right:
        st.markdown("**Candidate shots** (generated and simulated by code; the agents only choose)")
        st.dataframe(pd.DataFrame([{"id": c.shot_id, "shot": c.description} for c in menu]),
                     hide_index=True, use_container_width=True, height=330)
        if st.button("Run single agent and multi-agent", type="primary"):
            sa, ma = get_policies(mode)
            ctx = DecisionContext(state=state, candidates=menu)
            out = []
            for label, pol in (("Single agent", sa), ("Multi-agent", ma)):
                with st.spinner(f"{label} is deciding..."):
                    t0 = time.perf_counter()
                    try:
                        out.append(pol.decide(ctx))
                    except Exception as exc:  # noqa: BLE001 - show, never crash the demo
                        st.error(f"{label} failed: {type(exc).__name__}: {exc}")
                        out.append(None)
                    st.session_state.setdefault("timings", []).append(time.perf_counter() - t0)
            st.session_state.setdefault("decisions", {})[key] = tuple(out)
            st.rerun()

    if res:
        a, b = st.columns(2)
        with a:
            decision_card("Single agent", res[0], menu)
        with b:
            decision_card("Multi-agent (coordinator)", res[1], menu)
            if res[1] is not None:
                specialist_cards(res[1])
        if "record" in entry:
            with st.expander("Ground truth (oracle expected inning value)"):
                rec = entry["record"]
                rows = [{"id": k, "EIV": v["eiv"], "±SE": v["se"]} for k, v in rec["oracle"].items()]
                st.dataframe(pd.DataFrame(rows).sort_values("EIV", ascending=False), hide_index=True)
                for label, d in (("Single agent", res[0]), ("Multi-agent", res[1])):
                    if d is not None:
                        s = score_choice(rec, d.shot_id)
                        st.write(f"{label}: regret {s['regret']:.2f} (0 = best shot)")
        choices = [d.shot_id for d in res if d is not None]
        if choices:
            pick = st.radio("Execute which choice?", sorted(set(choices)), horizontal=True)
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
            st.markdown("**Break shot** (shared, seeded, not chosen by any agent)")
        else:
            d = r["decision"]
            st.markdown(f"**{r['system']} chose {d['shot_id']}** ({d['intent']})"
                        + (" — fallback" if r.get("fallback") else ""))
            ch = next((c for c in cands if c.shot_id == d["shot_id"]), None)
            if ch:
                st.caption(ch.description)
            st.write(d.get("rationale", ""))
            reports = (r.get("trace") or {}).get("reports") or {}
            for name, rep in reports.items():
                st.markdown(f"- *{name}*: top {rep.get('top_choice')} — {rep.get('summary', rep.get('error', ''))}")
        res = ("FOUL: " + o["foul"]) if o.get("foul") else (
            "potted " + ", ".join(o["potted"]) if o.get("potted") else "nothing potted")
        st.markdown(f"**Result:** {res}" + (" · game over, winner " + str(o['winner']) if o.get("game_over") else ""))
        if st.button("Animate this shot"):
            ex = r["executed"]
            traj = traj_for(state.model_dump_json(), (ex["phi"], ex["V0"], ex["a"], ex["b"], ex["theta"]))
            st.plotly_chart(animation_figure(traj, title="Executed stroke"), use_container_width=False)


def tab_results() -> None:
    found = False
    for p in sorted(glob.glob(str(ROOT / "results" / "*" / "summary.csv"))):
        found = True
        st.markdown(f"**{Path(p).parent.name}**")
        st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
    for p in sorted(glob.glob(str(ROOT / "results" / "figures" / "*.png"))):
        found = True
        st.image(p, caption=Path(p).stem)
    if not found:
        st.info("No results yet. Run the experiments (see README).")


def main() -> None:
    _warm()
    st.title("CueMind: do specialised LLM agents choose better pool shots than one LLM?")
    st.caption("Code generates and simulates candidate shots; the LLM systems only choose among them.")
    with st.sidebar:
        mode = st.radio("LLM mode", ["Offline (fake LLM)", "Claude (live)", "Claude (cache only)"])
        access = st.selectbox("Information given to the agents", ["L2", "L1", "L0"],
                              help="L0 geometry only · L1 one perfect-stroke preview · L2 statistics over realistic strokes")
        noise = st.selectbox("Execution noise", ["medium", "low", "high", "none"])
        if mode != "Offline (fake LLM)":
            st.caption("Live calls use Vertex AI credentials from .env; responses are cached.")
    t1, t2, t3 = st.tabs(["Decide a shot", "Replay a game", "Results"])
    with t1:
        tab_decide(mode, access, noise)
    with t2:
        tab_replay()
    with t3:
        tab_results()


main()
