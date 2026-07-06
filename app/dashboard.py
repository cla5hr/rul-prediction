"""Fleet dashboard for the RUL service.

Run with:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from rul_service.artifacts import bundle_exists
from rul_service.config import config
from rul_service.data import load_raw
from rul_service.drift import compute_drift
from rul_service.predict import Predictor

st.set_page_config(page_title="RUL Predictive Maintenance", page_icon="🛩️", layout="wide")
st.title("🛩️ Turbofan Predictive Maintenance")
st.caption(
    "Remaining-Useful-Life predictions with maintenance alerts and data-drift "
    "monitoring — NASA CMAPSS FD001."
)

STATUS_COLORS = {"healthy": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"}


@st.cache_resource
def _predictor() -> Predictor:
    return Predictor.from_dir(config.artifacts_dir, config)


@st.cache_data
def _test_engines() -> dict:
    test = load_raw(config.test_file)
    out = {}
    for eid in sorted(test["engine_id"].unique()):
        g = test[test["engine_id"] == eid].sort_values("cycle")
        out[str(eid)] = g.drop(columns=["engine_id", "cycle"]).to_dict(orient="records")
    return out


if not bundle_exists(config.artifacts_dir):
    st.error("No trained model found. Run `rul-service train` first.")
    st.stop()

pred = _predictor()
engines = _test_engines()

with st.sidebar:
    st.header("Model")
    st.metric("Test RMSE", pred.bundle.metrics.get("rmse"))
    st.metric("Test MAE", pred.bundle.metrics.get("mae"))
    st.caption(f"Architecture: `{pred.bundle.model_type}` · {pred.bundle.metrics.get('n_params'):,} params")
    st.caption(f"Features: {len(pred.bundle.preprocessor.feature_columns)}")

tab_fleet, tab_engine, tab_drift = st.tabs(["Fleet overview", "Single engine", "Drift monitor"])

# --------------------------------------------------------------- Fleet tab
with tab_fleet:
    results = [pred.predict(c, engine_id=eid, with_drift=False) for eid, c in engines.items()]
    df = pd.DataFrame(results).sort_values("rul")
    c1, c2, c3 = st.columns(3)
    c1.metric("Engines", len(df))
    c2.metric("⚠️ Warning", int((df["status"] == "warning").sum()))
    c3.metric("🔴 Critical", int((df["status"] == "critical").sum()))

    fig = px.bar(
        df, x="engine_id", y="rul", color="status",
        color_discrete_map=STATUS_COLORS, title="Predicted RUL per engine (sorted)",
    )
    fig.update_layout(xaxis_title="Engine", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Maintenance queue (lowest RUL first)")
    st.dataframe(
        df[["engine_id", "rul", "status"]].head(15), use_container_width=True, hide_index=True
    )

# -------------------------------------------------------------- Engine tab
with tab_engine:
    eid = st.selectbox("Engine", list(engines.keys()))
    cycles = engines[eid]
    res = pred.predict(cycles, engine_id=eid, with_drift=False)
    a, b, c = st.columns(3)
    a.metric("Predicted RUL", f"{res['rul']:.1f} cycles")
    b.metric("Status", res["status"].upper())
    c.metric("Cycles available", len(cycles))

    raw = pd.DataFrame(cycles)
    feats = [f for f in pred.bundle.preprocessor.feature_columns if f in raw.columns]
    sensor = st.selectbox("Inspect sensor trace", feats, index=min(3, len(feats) - 1))
    st.plotly_chart(
        px.line(raw.reset_index(), x="index", y=sensor, title=f"{sensor} over cycles"),
        use_container_width=True,
    )

# --------------------------------------------------------------- Drift tab
with tab_drift:
    st.write(
        "Population Stability Index (PSI) of recent data vs the training reference. "
        "Use the slider to simulate sensor drift (e.g. a miscalibrated sensor) and watch "
        "the detector react."
    )
    shift = st.slider("Injected sensor offset (simulated drift)", 0.0, 0.5, 0.0, 0.05)

    all_scaled = []
    for c in engines.values():
        all_scaled.append(pred._scaled_from_cycles(c))
    stacked = np.vstack(all_scaled)
    if shift > 0:
        stacked = stacked + shift  # uniform offset across features

    result = compute_drift(
        stacked, pred.bundle.reference, pred.bundle.preprocessor.feature_columns,
        warn=config.psi_warn, alert=config.psi_alert,
    )
    badge = {"ok": "🟢 OK", "warning": "🟡 WARNING", "alert": "🔴 ALERT",
             "insufficient_data": "⚪ insufficient data"}[result.status]
    st.subheader(f"Fleet drift status: {badge}  (max PSI = {result.max_psi:.3f})")

    psi_df = pd.DataFrame(
        sorted(result.per_feature_psi.items(), key=lambda kv: -kv[1]),
        columns=["feature", "psi"],
    )
    if not psi_df.empty:
        fig = px.bar(psi_df, x="feature", y="psi", title="PSI per feature")
        fig.add_hline(y=config.psi_warn, line_dash="dot", annotation_text="warn")
        fig.add_hline(y=config.psi_alert, line_dash="dash", annotation_text="alert")
        st.plotly_chart(fig, use_container_width=True)
