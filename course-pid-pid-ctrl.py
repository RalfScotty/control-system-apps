# === 0. PAGE CONFIG ===
import streamlit as st
st.set_page_config(page_title="PID Controller", page_icon="🕹️", layout="wide")

# === 1. IMPORTS ===
import numpy as np
import json
from dataclasses import dataclass
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 2. STYLING & CONSTANTS ===

_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    font=dict(color="#C9D1D9"),
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
_AX = dict(showgrid=True, gridcolor="#21262D", zeroline=True, zerolinecolor="#444")

CUSTOM_CSS = """
<style>
section[data-testid='stSidebar'] { min-width: 310px; max-width: 310px; }
button[data-baseweb='tab'] span { font-size: 18px; font-weight: 600; }
div[data-testid='stMetric'] {
    background: #0E1117; border: 1px solid #303030;
    border-radius: 8px; padding: 8px 14px;
}
</style>
"""

# Signal color coding per app guidelines
C_PV = "#00CC96"                    # Process variable — green
C_SP = "#EF553B"                    # Setpoint — red
C_OP = "#FF7F0E"                    # Controller output — orange
C_P  = "#58A6FF"                    # Proportional component
C_I  = "#A78BFA"                    # Integral component
C_D  = "#FB923C"                    # Derivative component

# === 3. SYSTEM PRESETS ===

SYSTEMS = {
    "PT1": {
        "label": "PT1 — First-Order Lag",
        "desc":  "Single pole · Kp=0.1 · T=5 s",
        "b0": 0.1, "a0": 1.0, "a1": 5.0, "a2": 0.0,
        "tend": 100.0, "ts": 0.1,
        "sp_max": 9.0,  "sp_default": 6.0,
        "kp": 3.0, "tn": 10.0, "tv": 2.0,
    },
    "IT1": {
        "label": "IT1 — Integrating + Lag",
        "desc":  "Integrating system · needs I-action",
        "b0": 0.5, "a0": 0.0, "a1": 1.0, "a2": 25.0,
        "tend": 300.0, "ts": 0.3,
        "sp_max": 50.0, "sp_default": 30.0,
        "kp": 0.3, "tn": 50.0, "tv": 10.0,
    },
    "PT2": {
        "label": "PT2 — Second-Order Lag (Overdamped)",
        "desc":  "T₁=40 s · T₂=200 s · Kp=3.5",
        "b0": 3.5, "a0": 1.0, "a1": 240.0, "a2": 8000.0,
        "tend": 1200.0, "ts": 1.2,
        "sp_max": 100.0, "sp_default": 50.0,
        "kp": 0.3, "tn": 200.0, "tv": 30.0,
    },
    "PT2 Osc": {
        "label": "PT2 Osc — Oscillating Second-Order",
        "desc":  "Underdamped · ζ≈0.1 · ω₀=0.1 rad/s",
        "b0": 20.3, "a0": 1.0, "a1": 2.0, "a2": 100.0,
        "tend": 200.0, "ts": 0.2,
        "sp_max": 50.0, "sp_default": 30.0,
        "kp": 0.05, "tn": 20.0, "tv": 5.0,
    },
}

DEFAULTS = {
    "system":              "PT1",
    "tdelay":              0.0,
    "ctrl_type":           "PID",
    "kp":                  3.0,
    "tn":                  10.0,
    "tv":                  2.0,
    "op_min":              0.0,
    "op_max":              100.0,
    "anti_windup":         True,
    "sp_value":            6.0,
    "sp2_enable":          False,
    "sp2_value":           3.0,
    "dist_enable":         False,
    "dist_value":          20.0,
}

# === 4. STATE MANAGEMENT ===

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def on_system_change():
    s = SYSTEMS[st.session_state["system"]]
    st.session_state["kp"]       = s["kp"]
    st.session_state["tn"]       = s["tn"]
    st.session_state["tv"]       = s["tv"]
    st.session_state["sp_value"] = s["sp_default"]
    st.session_state["sp2_value"]= round(s["sp_default"] * 0.5, 1)

def on_preset_fast():
    st.session_state["kp"] = round(st.session_state["kp"] * 1.5, 3)
    st.session_state["tn"] = round(st.session_state["tn"] * 0.7, 1)
    st.session_state["tv"] = round(st.session_state["tv"] * 1.3, 1)

def on_preset_slow():
    st.session_state["kp"] = round(st.session_state["kp"] * 0.5, 3)
    st.session_state["tn"] = round(st.session_state["tn"] * 1.5, 1)
    st.session_state["tv"] = round(st.session_state["tv"] * 0.7, 1)

def on_preset_reset():
    s = SYSTEMS[st.session_state["system"]]
    st.session_state["kp"] = s["kp"]
    st.session_state["tn"] = s["tn"]
    st.session_state["tv"] = s["tv"]

def on_upload():
    f = st.session_state.get("cfg_upload")
    if f is None:
        return
    try:
        cfg = json.loads(f.read().decode())
        for k, v in cfg.items():
            if k in DEFAULTS:
                st.session_state[k] = type(DEFAULTS[k])(v)
        st.toast("Configuration loaded ✓")
    except Exception:
        st.error("Invalid configuration file.")

# === 5. SIMULATION ENGINE ===

@dataclass(frozen=True)
class SimCfg:
    b0: float;  a0: float;  a1: float;  a2: float
    tdelay: float
    ctrl_type: str
    kp: float;  tn: float;  tv: float;  tf: float
    anti_windup: bool
    op_min: float;  op_max: float
    tend: float;  ts: float
    sp_value: float
    sp2_enable: bool;  sp2_value: float
    dist_enable: bool;  dist_value: float

def _ode(x, _t, u, b0, a0, a1, a2):
    y, dy = x[0], x[1]
    if a2 == 0.0:
        return [(-a0 * y + b0 * u) / a1, 0.0]
    return [dy, (-a1 * dy - a0 * y + b0 * u) / a2]

@st.cache_data
def simulate(cfg: SimCfg):
    t  = np.arange(0, cfg.tend + cfg.ts, cfg.ts)
    ns = len(t) - 1
    dt = cfg.ts

    # Setpoint profile
    sp = np.zeros(ns + 1)
    i0 = max(1, int(0.05 * ns))
    sp[i0:] = cfg.sp_value
    if cfg.sp2_enable:
        sp[int(0.6 * ns):] = cfg.sp2_value

    # Disturbance pulse (5% duration at t=40%)
    dist = np.zeros(ns + 1)
    if cfg.dist_enable:
        da, db = int(0.40 * ns), int(0.45 * ns)
        dist[da:db] = cfg.dist_value

    # Derivative filter coefficient
    tf = cfg.tf
    tv_eff = cfg.tv * (tf / dt * (1.0 - np.exp(-dt / tf))) if tf > 1e-9 else cfg.tv

    ndelay = int(np.ceil(cfg.tdelay / dt))
    has_I  = cfg.ctrl_type in ("PI", "PID")
    has_D  = cfg.ctrl_type in ("PD", "PID")

    op    = np.zeros(ns + 1)
    pv    = np.zeros((ns + 1, 2))
    e     = np.zeros(ns + 1)
    ie    = np.zeros(ns + 1)
    P_arr = np.zeros(ns + 1)
    I_arr = np.zeros(ns + 1)
    D_arr = np.zeros(ns + 1)
    SAT   = np.zeros(ns + 1)

    for i in range(ns):
        e[i] = sp[i] - pv[i, 0]

        P_arr[i] = cfg.kp * e[i]

        if has_I:
            ie[i]    = (ie[i - 1] + e[i] * dt) if i > 0 else e[i] * dt
            I_arr[i] = cfg.kp / cfg.tn * ie[i]

        if has_D and tf > 1e-9:
            if i > 0:
                D_arr[i] = (cfg.kp * tv_eff / tf * (e[i] - e[i - 1])
                            + D_arr[i - 1] * np.exp(-dt / tf))
            else:
                D_arr[i] = cfg.kp * tv_eff / tf * e[i]

        op[i] = P_arr[i] + I_arr[i] + D_arr[i]

        # Anti-windup: undo last integral step when saturated in error direction
        if cfg.anti_windup and has_I:
            if op[i] > cfg.op_max and e[i] > 0:
                op[i]    = cfg.op_max
                ie[i]   -= e[i] * dt
                I_arr[i] = cfg.kp / cfg.tn * ie[i]
                SAT[i]   = cfg.op_max
            elif op[i] < cfg.op_min and e[i] < 0:
                op[i]    = cfg.op_min
                ie[i]   -= e[i] * dt
                I_arr[i] = cfg.kp / cfg.tn * ie[i]
                SAT[i]   = cfg.op_min

        # Hard saturation
        if op[i] > cfg.op_max:
            op[i] = cfg.op_max;  SAT[i] = cfg.op_max
        elif op[i] < cfg.op_min:
            op[i] = cfg.op_min;  SAT[i] = cfg.op_min

        iop   = max(0, i - ndelay)
        u_in  = op[iop] + dist[i]
        y_new = odeint(_ode, pv[i], [0, dt],
                       args=(u_in, cfg.b0, cfg.a0, cfg.a1, cfg.a2))
        pv[i + 1, 0] = y_new[-1, 0]
        pv[i + 1, 1] = y_new[-1, 1]

    for arr in (op, P_arr, I_arr, D_arr):
        arr[ns] = arr[ns - 1]

    return t, pv, op, sp, e, P_arr, I_arr, D_arr, SAT

# === 6. PERFORMANCE METRICS ===

def compute_metrics(t, pv_y, sp, e, sp_target):
    dt       = t[1] - t[0]
    sp_start = max(1, int(0.05 * len(t)))
    pv_a     = pv_y[sp_start:]
    t_a      = t[sp_start:] - t[sp_start]
    e_a      = e[sp_start:]

    # Overshoot
    if sp_target != 0:
        overshoot = max(0.0, (np.max(pv_a) - sp_target) / abs(sp_target) * 100.0)
    else:
        overshoot = 0.0

    # Settling time: last time pv is outside ±2% band
    band         = 0.02 * abs(sp_target) if sp_target != 0 else 0.02
    settling_time = float("nan")
    for j in range(len(pv_a) - 1, -1, -1):
        if abs(pv_a[j] - sp_target) > band:
            idx          = min(j + 1, len(pv_a) - 1)
            settling_time = float(t_a[idx])
            break
    else:
        settling_time = 0.0

    # Steady-state error (last 15%)
    tail     = pv_a[int(0.85 * len(pv_a)):]
    ss_error = float(abs(sp_target - np.mean(tail))) if len(tail) > 0 else float("nan")

    ise  = float(np.sum(e_a ** 2) * dt)
    itae = float(np.sum(t_a * np.abs(e_a)) * dt)

    return {"overshoot": overshoot, "settling_time": settling_time,
            "ss_error": ss_error, "ise": ise, "itae": itae}

# === 7. PLOTTING ===

def make_main_figure(t, pv_y, op, sp, SAT):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Process Variable & Setpoint", "Controller Output (OP)"],
    )

    # ±2 % tolerance band
    band = 0.02 * (np.max(np.abs(sp)) or 1)
    fig.add_trace(go.Scatter(
        x=t, y=sp + band, mode="lines",
        line=dict(color="rgba(239,85,59,0.0)", width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=sp - band, mode="lines",
        fill="tonexty", fillcolor="rgba(239,85,59,0.08)",
        line=dict(color="rgba(239,85,59,0.0)", width=0),
        name="±2 % Band", hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=sp, mode="lines", name="Setpoint (SP)",
        line=dict(color=C_SP, width=2, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=pv_y, mode="lines", name="Process Variable (PV)",
        line=dict(color=C_PV, width=2.5),
    ), row=1, col=1)

    # Anti-windup active zones
    if np.any(SAT != 0):
        aw_y = np.where(SAT != 0, op, np.nan)
        fig.add_trace(go.Scatter(
            x=t, y=aw_y, mode="lines",
            line=dict(color="rgba(168,139,250,0.6)", width=4),
            name="Anti-Windup Active",
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=op, mode="lines", name="Controller Output (OP)",
        line=dict(color=C_OP, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.08)",
    ), row=2, col=1)

    fig.update_layout(height=520, **_BASE)
    fig.update_xaxes(**_AX, title_text="", row=1, col=1)
    fig.update_xaxes(**_AX, title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(**_AX, title_text="Process Value", row=1, col=1)
    fig.update_yaxes(**_AX, title_text="Output", row=2, col=1)
    return fig

def make_components_figure(t, P_arr, I_arr, D_arr, ctrl_type):
    traces = []
    if True:
        traces.append(("P", P_arr, C_P))
    if ctrl_type in ("PI", "PID"):
        traces.append(("I", I_arr, C_I))
    if ctrl_type in ("PD", "PID"):
        traces.append(("D", D_arr, C_D))

    n   = len(traces)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"{name} Component" for name, _, _ in traces],
    )
    for idx, (name, arr, color) in enumerate(traces, 1):
        fig.add_trace(go.Scatter(
            x=t, y=arr, mode="lines", name=f"{name} Part",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if color.startswith("rgb") else color + "14",
        ), row=idx, col=1)

    fig.update_layout(height=100 + 140 * n, showlegend=False, **_BASE)
    for idx in range(1, n + 1):
        fig.update_xaxes(**_AX, row=idx, col=1)
        fig.update_yaxes(**_AX, row=idx, col=1)
    fig.update_xaxes(title_text="Time [s]", row=n, col=1)
    return fig

# === 8. SIDEBAR ===

def render_sidebar():
    with st.sidebar:
        st.title("🕹️ PID Control Lab")
        st.caption("Interactive PID tuning sandbox")
        st.divider()

        # --- Plant ---
        st.subheader("🏗️ Plant")
        sys_key = st.selectbox(
            "System Type",
            options=list(SYSTEMS.keys()),
            format_func=lambda k: SYSTEMS[k]["label"],
            key="system",
            on_change=on_system_change,
        )
        sys = SYSTEMS[sys_key]
        st.caption(sys["desc"])
        st.slider("Time Delay [s]", 0.0, 20.0, step=0.5, key="tdelay",
                  help="Transport delay applied to the process input")
        st.divider()

        # --- Setpoint ---
        st.subheader("🎯 Setpoint")
        sp_max = float(sys["sp_max"])
        st.slider("SP₁ (step at t=5 %)", 0.0, sp_max,
                  step=round(sp_max / 50, 2), key="sp_value",
                  help="First setpoint step applied at 5 % of simulation time")
        sp2_on = st.checkbox("Second SP step (at t=60 %)", key="sp2_enable")
        st.slider("SP₂", 0.0, sp_max,
                  step=round(sp_max / 50, 2), key="sp2_value",
                  disabled=not sp2_on,
                  help="Second setpoint applied at 60 % of simulation time")
        dist_on = st.checkbox("Disturbance pulse (at t=40 %)", key="dist_enable")
        st.slider("Disturbance magnitude", -50.0, 50.0, step=1.0,
                  key="dist_value", disabled=not dist_on,
                  help="Pulse added to the process input for 5 % of simulation time")
        st.divider()

        # --- Controller ---
        st.subheader("📈 Controller")
        ctrl = st.radio("Type", ["P", "PI", "PD", "PID"],
                        horizontal=True, key="ctrl_type")
        has_I = ctrl in ("PI", "PID")
        has_D = ctrl in ("PD", "PID")

        st.number_input("Kp — Proportional Gain", min_value=0.001, max_value=500.0,
                        step=0.001, format="%.3f", key="kp",
                        help="Scales all three controller terms")
        st.number_input("Tn — Integration Time [s]", min_value=0.1, max_value=2000.0,
                        step=0.1, format="%.1f", key="tn",
                        disabled=not has_I,
                        help="Larger Tn → slower integral action")
        st.number_input("Tv — Derivative Time [s]", min_value=0.1, max_value=500.0,
                        step=0.1, format="%.1f", key="tv",
                        disabled=not has_D,
                        help="Larger Tv → stronger derivative kick")
        st.divider()

        # --- Output Limits ---
        st.subheader("⚙️ Output Limits")
        c1, c2 = st.columns(2)
        c1.number_input("Min", step=10.0, format="%.1f", key="op_min")
        c2.number_input("Max", step=10.0, format="%.1f", key="op_max")
        st.checkbox("Anti-Windup", key="anti_windup",
                    disabled=not has_I,
                    help="Prevents integral windup during output saturation")
        st.divider()

        # --- Presets ---
        st.subheader("⚡ Presets")
        ca, cb, cc = st.columns(3)
        ca.button("⚡ Fast",   on_click=on_preset_fast,  use_container_width=True,
                  help="Kp ×1.5 · Tn ×0.7 · Tv ×1.3")
        cb.button("🐢 Slow",  on_click=on_preset_slow,  use_container_width=True,
                  help="Kp ×0.5 · Tn ×1.5 · Tv ×0.7")
        cc.button("↺ Reset",  on_click=on_preset_reset, use_container_width=True,
                  help="Restore system defaults")
        st.divider()

        # --- Save / Load ---
        st.subheader("💾 Save / Load Config")
        cfg_dict = {k: st.session_state.get(k, DEFAULTS[k]) for k in DEFAULTS}
        st.download_button(
            "⬇️ Download Config", data=json.dumps(cfg_dict, indent=2),
            file_name="pid_config.json", mime="application/json",
            use_container_width=True,
        )
        st.file_uploader("Upload Config", type="json",
                         label_visibility="collapsed",
                         key="cfg_upload", on_change=on_upload)

# === 9. TAB: SIMULATION ===

def render_tab_simulation(t, pv, op, sp, e, P_arr, I_arr, D_arr, SAT, metrics, cfg):
    pv_y = pv[:, 0]

    # --- Metrics row ---
    m = metrics
    os_str  = f"{m['overshoot']:.1f} %"
    ts_str  = f"{m['settling_time']:.1f} s" if not np.isnan(m['settling_time']) else "—"
    sse_str = f"{m['ss_error']:.3f}"
    ise_str = f"{m['ise']:.1f}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overshoot",          os_str,
              help="Peak deviation above the setpoint, relative to SP")
    c2.metric("Settling Time (2 %)", ts_str,
              help="Time from step until PV stays within ±2 % of SP")
    c3.metric("Steady-State Error", sse_str,
              help="Mean absolute error over the last 15 % of simulation")
    c4.metric("ISE",                ise_str,
              help="Integral Squared Error — penalises large deviations")

    # --- Main chart ---
    st.plotly_chart(make_main_figure(t, pv_y, op, sp, SAT),
                    use_container_width=True)

    # --- CSV download ---
    import pandas as pd
    df = pd.DataFrame({
        "t [s]": t, "SP": sp, "PV": pv_y,
        "OP": op, "Error e": e,
        "P": P_arr, "I": I_arr, "D": D_arr,
    })
    st.download_button(
        "⬇️ Download CSV", data=df.to_csv(index=False),
        file_name=f"pid_sim_{cfg.ctrl_type}_{st.session_state['system']}.csv",
        mime="text/csv",
    )

    # --- P/I/D Components ---
    with st.expander("📊 P / I / D Components", expanded=False):
        st.plotly_chart(make_components_figure(t, P_arr, I_arr, D_arr, cfg.ctrl_type),
                        use_container_width=True)

    # --- Anti-windup status ---
    if cfg.anti_windup and np.any(SAT != 0):
        st.warning(
            f"**Anti-Windup active** — output was saturated at "
            f"{int(np.max(np.abs(SAT[SAT != 0])))} for "
            f"{int(np.sum(SAT != 0)) * cfg.ts:.1f} s total.",
            icon="⚠️",
        )

# === 10. TAB: THEORY ===

def render_tab_theory():
    st.subheader("PID Controller — Parallel Form")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
The **PID controller** computes an output signal $u(t)$ from the control error
$e(t) = r(t) - y(t)$ by summing three terms:
""")
        st.latex(r"u(t) = K_p \Bigl[ e(t) + \frac{1}{T_n}\int_0^t e(\tau)\,d\tau + T_v\,\dot{e}(t) \Bigr]")

        st.markdown("**Transfer function (parallel / ideal form):**")
        st.latex(r"C(s) = K_p \left(1 + \frac{1}{T_n s} + \frac{T_v s}{T_f s + 1}\right)")

        st.markdown("""
| Symbol | Name | Effect |
|--------|------|--------|
| $K_p$ | Proportional gain | Scales all three terms; higher → faster but more oscillation |
| $T_n$ | Integration time | Lower → stronger I action; **eliminates steady-state error** |
| $T_v$ | Derivative time | Higher → stronger D action; **anticipates** error changes |
| $T_f$ | Filter time ($≈T_v/10$) | Low-pass filter on derivative to suppress noise |
""")

    with col2:
        st.markdown("**Discrete derivative with filter:**")
        st.latex(r"""
D_i = K_p\,\frac{T_{v,\text{eff}}}{T_f}(e_i - e_{i-1})
      + D_{i-1}\,e^{-\Delta t / T_f}
""")
        st.markdown("**Discrete integrator (forward Euler):**")
        st.latex(r"\mathrm{ie}_i = \mathrm{ie}_{i-1} + e_i \cdot \Delta t")
        st.latex(r"I_i = \frac{K_p}{T_n} \cdot \mathrm{ie}_i")

    st.divider()

    # --- Controller types ---
    st.subheader("Controller Types")
    ca, cb, cc, cd = st.columns(4)
    ca.info("**P-Controller**\n\nProportional only. Fast but leaves a permanent steady-state error (offset). Useful for processes without integrating behaviour.")
    cb.info("**PI-Controller**\n\nAdds integral action → zero steady-state error. Most common in industry. Can cause slow oscillation if $T_n$ is too small.")
    cc.info("**PD-Controller**\n\nAdds predictive derivative action → reduces overshoot. No integral → steady-state error remains. Used when process is self-integrating.")
    cd.info("**PID-Controller**\n\nCombines all three. Eliminates steady-state error and reduces overshoot. Derivative term requires noise filtering.")

    st.divider()

    # --- Anti-Windup ---
    st.subheader("Anti-Windup Mechanism")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("""
**Problem:** When the controller output is saturated (e.g. $u = u_\\text{max}$) but the error
$e(t)$ remains positive, the integral term continues to grow — this is called **integral windup**.

When the error finally changes sign, the integrator must first unwind before $u$ can decrease,
causing a **large overshoot and slow recovery**.
""")
    with col2:
        st.markdown("""
**Solution (back-calculation / clamping):**

When saturation is detected in the direction of the current error, undo the last
integration step:
""")
        st.latex(r"""
\text{if } u_i > u_{\max}\ \text{and}\ e_i > 0:
\quad \mathrm{ie}_i \leftarrow \mathrm{ie}_i - e_i \cdot \Delta t
""")
        st.markdown("""
This keeps the integrator state near the saturation boundary, so recovery
is immediate when the error changes sign.
""")

    st.divider()

    # --- Process models ---
    st.subheader("Process Models")
    pa, pb, pc, pd = st.columns(4)
    pa.markdown("**PT1** — First-Order Lag")
    pa.latex(r"G(s) = \frac{K_p}{T_1 s + 1}")
    pa.caption("Kp=0.1, T₁=5 s")

    pb.markdown("**IT1** — Integrating + Lag")
    pb.latex(r"G(s) = \frac{K_p}{s\,(T_1 s + 1)}")
    pb.caption("Integrating — no steady-state without I-action")

    pc.markdown("**PT2** — 2nd-Order Overdamped")
    pc.latex(r"G(s) = \frac{K_p}{(T_1 s+1)(T_2 s+1)}")
    pc.caption("T₁=40 s, T₂=200 s")

    pd.markdown("**PT2 Osc** — 2nd-Order Underdamped")
    pd.latex(r"G(s) = \frac{\omega_0^2}{s^2 + 2\zeta\omega_0 s + \omega_0^2}")
    pd.caption("ζ≈0.1, ω₀=0.1 rad/s — prone to oscillation")

# === 11. TAB: EXERCISES ===

def render_tab_exercises():
    st.subheader("Exercises — PID Controller Tuning")
    st.markdown("Use the sidebar controls to explore each scenario, then check your answers below.")
    st.divider()

    # --- Exercise 1 ---
    st.markdown("### Exercise 1 — P-Controller on a PT1 Plant")
    st.markdown("""
Select system **PT1** and controller type **P**. Set $K_p = 3$, SP₁ = 6.

- **(a)** What steady-state error do you observe? Why does the P-controller not reach zero error?
- **(b)** Increase $K_p$ to 10. Does the error improve? What is the trade-off?
- **(c)** Switch to PI. What happens to the steady-state error?
""")
    with st.expander("🔍 Solution (a)"):
        st.success("Steady-state error ≈ 1.5 units (PV settles below SP = 6).")
    with st.expander("💡 Explanation (a)"):
        st.markdown(r"""
For a PT1 plant with gain $K_{p,\text{proc}}$ and a P-controller with gain $K_p$, the closed-loop
steady-state error is:

$$e_\infty = \frac{r}{1 + K_p \cdot K_{p,\text{proc}}}$$

With $K_{p,\text{proc}} = 0.1$ and $K_p = 3$: $e_\infty = 6 / (1 + 0.3) \approx 1.54$.

A P-controller alone **cannot eliminate steady-state error** because it needs a non-zero error
to produce any output. The integral term in a PI controller removes this offset entirely.
""")
    with st.expander("🔍 Solution (b)"):
        st.success("Higher Kp reduces the error but causes the response to become faster and may overshoot.")
    with st.expander("💡 Explanation (b)"):
        st.markdown(r"""
Increasing $K_p$ reduces $e_\infty$ (since it appears in the denominator) but also increases
the loop gain, which can lead to oscillation or instability depending on the plant dynamics.
This is the fundamental **gain-bandwidth trade-off** in control design.
""")

    st.divider()

    # --- Exercise 2 ---
    st.markdown("### Exercise 2 — Integrating System (IT1)")
    st.markdown("""
Select system **IT1** and controller type **P**. Set $K_p = 0.3$, SP₁ = 30.

- **(a)** Observe the response. Does the PV reach the setpoint?
- **(b)** Switch to PI with $T_n = 50$. What changes?
- **(c)** Enable Anti-Windup and disable it. Compare the overshoot after the initial transient.
""")
    with st.expander("🔍 Solution (a)"):
        st.success("With P-only control, the IT1 process can reach SP but may oscillate or drift — the system itself contains an integrator.")
    with st.expander("💡 Explanation (a)"):
        st.markdown(r"""
The IT1 process already contains an integrator: $G(s) = \frac{0.5}{s(25s+1)}$.

A P-controller creates a Type-1 open-loop system (one integrator), which means
**zero steady-state error for step setpoints** is theoretically possible. However, the
phase margin may be low, causing oscillation.

Adding I-action creates a Type-2 loop, which also eliminates ramp errors — but
makes stability harder to achieve without derivative damping.
""")
    with st.expander("🔍 Solution (c)"):
        st.success("Without Anti-Windup, the initial overshoot is significantly larger because the integrator winds up during saturation.")
    with st.expander("💡 Explanation (c)"):
        st.markdown(r"""
During the initial transient, the large error causes the integral state to accumulate beyond
what is needed for steady-state. When the PV approaches SP, the integrator must unwind,
which delays the reduction of $u(t)$ and causes **overshoot**.

Anti-Windup prevents accumulation when $u$ is saturated, so the integrator stays near its
correct steady-state value from the beginning.
""")

    st.divider()

    # --- Exercise 3 ---
    st.markdown("### Exercise 3 — Derivative Action on an Oscillating Plant")
    st.markdown("""
Select system **PT2 Osc** and controller type **PI**. Set $K_p = 0.05$, $T_n = 20$.

- **(a)** Enable the **Disturbance pulse** (at t=40 %). How quickly does the controller recover?
- **(b)** Switch to **PID** with $T_v = 5$. Does recovery improve?
- **(c)** Increase $T_v$ to 20. What happens?
""")
    with st.expander("🔍 Solution (b)"):
        st.success("PID recovers faster and with less oscillation after the disturbance.")
    with st.expander("💡 Explanation (b)"):
        st.markdown(r"""
The derivative term reacts to the **rate of change** of the error. When a disturbance
hits, $\dot{e}$ is large, so the D-term immediately provides a large corrective kick —
much faster than the integrator can respond.

This makes PID especially effective for **disturbance rejection** in oscillating or
slow plants.
""")
    with st.expander("🔍 Solution (c)"):
        st.success("Too large Tv amplifies noise and can destabilise the loop.")
    with st.expander("💡 Explanation (c)"):
        st.markdown(r"""
The derivative filter time $T_f \approx T_v / 10$ limits high-frequency amplification.
But if $T_v$ becomes very large, even the filtered derivative becomes aggressive,
leading to **chattering or instability**.

A practical rule: $T_v$ should be 0.1–0.3 × the dominant process time constant.
""")

    st.divider()

    # --- Exercise 4 ---
    st.markdown("### Exercise 4 — Ziegler–Nichols Tuning on PT2")
    st.markdown("""
Select system **PT2** (slow, overdamped). Increase $K_p$ with a **P-controller** until
the closed-loop response begins to oscillate (the *ultimate gain* $K_{u}$).
Read off the oscillation period $T_u$ from the chart and compute PID parameters:

$$K_p = 0.6\\,K_u, \\quad T_n = 0.5\\,T_u, \\quad T_v = 0.125\\,T_u$$

Apply these values and check the response quality.
""")
    with st.expander("💡 Explanation — Ziegler–Nichols Method"):
        st.markdown(r"""
The **Ziegler–Nichols closed-loop method** finds the stability boundary empirically:

1. Set controller to **P-only** with $T_n = \infty$, $T_v = 0$.
2. Increase $K_p$ until the loop sustains **undamped oscillations** → this is $K_u$.
3. Measure the oscillation period $T_u$.
4. Apply the ZN tuning formulas above.

**Result:** The ZN method typically gives ~25 % overshoot — a reasonable starting point
that can then be refined manually or via optimisation.

**Limitations:** ZN requires reaching the stability boundary, which may not be
acceptable for real plants. Modern alternatives include SIMC or lambda tuning.
""")

# === 12. MAIN ===

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    ss  = st.session_state
    sys = SYSTEMS[ss["system"]]
    tf  = max(ss["tv"] / 10.0, 1e-6)

    cfg = SimCfg(
        b0=sys["b0"], a0=sys["a0"], a1=sys["a1"], a2=sys["a2"],
        tdelay=ss["tdelay"],
        ctrl_type=ss["ctrl_type"],
        kp=ss["kp"], tn=max(ss["tn"], 0.01), tv=ss["tv"], tf=tf,
        anti_windup=ss["anti_windup"],
        op_min=ss["op_min"], op_max=ss["op_max"],
        tend=sys["tend"], ts=sys["ts"],
        sp_value=ss["sp_value"],
        sp2_enable=ss["sp2_enable"], sp2_value=ss["sp2_value"],
        dist_enable=ss["dist_enable"], dist_value=ss["dist_value"],
    )

    t, pv, op, sp, e, P_arr, I_arr, D_arr, SAT = simulate(cfg)
    metrics = compute_metrics(t, pv[:, 0], sp, e, cfg.sp_value)

    tab_sim, tab_theory, tab_exercises = st.tabs([
        "🕹️ Simulation", "📘 Theory", "📝 Exercises"
    ])

    with tab_sim:
        render_tab_simulation(t, pv, op, sp, e, P_arr, I_arr, D_arr, SAT, metrics, cfg)
    with tab_theory:
        render_tab_theory()
    with tab_exercises:
        render_tab_exercises()

main()
