"""Community Cloud entry point: streamlit run streamlit_app.py."""

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from cloud_runner import PUBLIC_PROBLEMS, run_public
from dashboard_service import PROBLEMS

st.set_page_config(page_title="QAOA Query Optimization Lab", page_icon="⚛️", layout="wide")
st.title("QAOA Query Optimization Lab")
st.caption("Hybrid quantum–classical optimization · Python + PennyLane · ideal simulation")
st.info("A research prototype, not a production SQL optimizer. These small examples do not demonstrate quantum advantage.")

with st.sidebar:
    st.header("Experiment settings")
    with st.form("experiment"):
        problem = st.selectbox("Problem", PUBLIC_PROBLEMS, format_func=lambda key: PROBLEMS[key]["name"])
        method = st.selectbox("Classical optimizer", ("L-BFGS-B", "COBYLA", "ADAM"))
        depth = st.slider("QAOA depth", 1, 2, 2)
        budget = st.slider("Optimizer budget", 10, 100, 50, step=10)
        shots = st.select_slider("Simulated samples", (100, 500, 1000, 2000), value=1000)
        seed = st.number_input("Training seed", 0, 2**32-1, 42, step=1)
        sample_seed = st.number_input("Sampling seed", 0, 2**32-1, 123, step=1)
        penalty = st.selectbox("Join penalty (three-table mode only)", ("max_plus_one", "twice_max"))
        submitted = st.form_submit_button("Run experiment", type="primary")
    st.caption("Shared demo: one run at a time, two-minute timeout. The larger 10-qubit experiments remain available in the local dashboard.")
    st.link_button("Source code & research notes", "https://github.com/kaushickj2008/qaoa-query-optimization-lab")

st.write(PROBLEMS[problem]["description"])
st.caption("COBYLA budgets function evaluations; Adam budgets updates; L-BFGS-B budgets iterations. These budgets are not equivalent.")

if submitted:
    if time.monotonic() - st.session_state.get("last_started", -1e9) < 10:
        st.warning("Please wait ten seconds between starting experiments.")
    else:
        st.session_state.last_started = time.monotonic()
        config = dict(problem=problem, method=method, p=depth, maxiter=budget,
                      shots=shots, seed=int(seed), sample_seed=int(sample_seed), penalty=penalty)
        try:
            with st.spinner("Training and evaluating the quantum circuit…"):
                st.session_state.result = run_public(config)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

result = st.session_state.get("result")
if result is None:
    st.subheader("Explore an experiment")
    st.write("Choose a problem in the sidebar and click **Run experiment**. Compare its solution probabilities with the exact classical optimum, inspect training, and download the full result.")
else:
    st.subheader(result["problem_name"])
    cfg, metrics, training = result["config"], result["metrics"], result["training"]
    st.caption(f"Displayed result: p={cfg['p']} · {cfg['method']} · seed={cfg['seed']} · {cfg['shots']} samples · {result['elapsed_seconds']:.2f}s simulator time (excludes process startup)")
    cols = st.columns(4)
    cols[0].metric("Optimal probability", f"{100*metrics['optimal_probability']:.2f}%")
    cols[1].metric("Valid probability", f"{100*metrics['valid_probability']:.2f}%")
    expected = metrics["expected_value"]
    cols[2].metric(metrics["expected_label"], "N/A" if expected is None else f"{expected:.3f}")
    cols[3].metric("Exact classical optimum", f"{metrics['exact_value']:.3f}")
    st.caption(f"Observed optimal samples: {metrics['optimal_samples']} / {cfg['shots']}. Uniform valid-solution optimal probability: {100*metrics['uniform_valid_optimal_probability']:.2f}%.")
    st.subheader("Training energy")
    history = np.asarray(training["energy_history"])
    frame = pd.DataFrame({"Objective": history, "Best observed": np.minimum.accumulate(history)},
                         index=pd.RangeIndex(1, len(history)+1, name="Objective evaluation"))
    st.line_chart(frame)
    st.caption(f"{len(history)} objective calls · {training['gradient_evaluations']} gradient calls. Lower energy is better; objective calls are not optimizer iterations or hardware shots.")
    st.write(f"Optimizer stopping criterion reached: {training['success']}. {training['message']} This does not prove a global optimum.")
    st.subheader("Solution distribution")
    outcomes = pd.DataFrame(result["distribution"]).sort_values("probability", ascending=False)
    st.bar_chart(outcomes.set_index("bits")[["probability"]])
    st.dataframe(outcomes, hide_index=True)
    if cfg["problem"].endswith("_valid"):
        st.warning("The valid-plan mode classically enumerates and costs all 15 plans first. That preprocessing already solves the problem; this is a feasibility-preservation experiment, not a scalable speedup.")
        st.write(f"Exact plan: {metrics['exact_plan']}")
    elif cfg["problem"] == "join3":
        st.caption("Retained join costs exclude the constant root cost. Expected cost is conditional on validity; training uses penalty-normalized QUBO energy.")
    with st.expander("Parameters, configuration and model"):
        st.json({"config": cfg, "parameters": training["parameters"], "model": result["model"], "versions": result["versions"]})
    st.download_button("Download full result (JSON)", json.dumps(result, indent=2),
                       file_name=f"qaoa-{cfg['problem']}-seed{cfg['seed']}.json", mime="application/json")

st.divider()
st.caption("No real SQL execution or quantum hardware. Samples come from ideal statevector probabilities. Results are session-local and disappear when the session ends. No accounts or private input data are required.")
