# === 0. PAGE CONFIG ===
import streamlit as st
st.set_page_config(page_title="Level Controller", layout="wide", page_icon="🪣")

# === 1. IMPORTS ===
import numpy as np
import json
import io
from dataclasses import dataclass
from enum import IntEnum
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 2. CONSTANTS & STYLING ===

# Plotly base layout dicts
_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    font=dict(color="#C9D1D9"),
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
_AX = dict(showgrid=True, gridcolor="#21262D", zeroline=True, zerolinecolor="#444")

CUSTOM_CSS = """
<style>
section[data-testid='stSidebar'] { min-width: 310px; max-width: 310px; }
button[data-baseweb='tab'] span { font-size: 18px; font-weight: 600; }
div[data-testid='stMetric'] {
    background: #0E1117; border: 1px solid #303030;
    border-radius: 8px; padding: 10px 16px;
}
</style>
"""

# Signal colors
C_PV = "#00CC96"
C_SP = "red"
C_OP = "orange"

# === 3. DATA CLASSES & ENUMS ===

@dataclass(frozen=True)
class PlantConfig:
    V: float       # Process gain
    T: float       # Time constant (s)
    xi: float      # Damping factor
    Tdelay: float  # Dead time (s)

@dataclass(frozen=True)
class CtrlConfig:
    hys: float     # Hysteresis band (±)
    sp1: float     # Set point phase 1
    sp2: float     # Set point phase 2
    sp3: float     # Set point phase 3
    t1: float      # Time of SP step 1→2 (s)
    t2: float      # Time of SP step 2→3 (s)
    Tend: float    # Simulation end time (s)
    Ts: float      # Sample time (s)

class CtrlState(IntEnum):
    HIGH = 1
    LOW  = 2

# === 4. DEFAULTS & STATE MANAGEMENT ===

DEFAULTS = dict(
    V=4.0, T=80.0, xi=3.0, Tdelay=5.0,
    hys=0.5,
    sp1=100.0, sp2=150.0, sp3=0.0,
    t1=500.0, t2=750.0,
    Tend=1000.0, Ts=0.5,
)

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def safe_get(key):
    return st.session_state.get(key, DEFAULTS[key])

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# === 5. SIMULATION ===

def _process(x, t, u, V, T, xi):
    """Second-order ODE: T² y'' + 2ξT y' + y = V·u"""
    y, dydt = x
    dy2dt2 = (-2.0 * xi * T * dydt - y + V * u) / T**2
    return [dydt, dy2dt2]

@st.cache_data
def simulate(plant: PlantConfig, ctrl: CtrlConfig):
    Tend, Ts = ctrl.Tend, ctrl.Ts
    t = np.arange(0, Tend + Ts, Ts)
    ns = len(t) - 1

    # Build set-point array
    sp = np.full(ns + 1, ctrl.sp1)
    sp[int(ctrl.t1 / Ts):] = ctrl.sp2
    sp[int(ctrl.t2 / Ts):] = ctrl.sp3

    ndelay = int(np.ceil(plant.Tdelay / Ts))
    op  = np.zeros(ns + 1)
    pv  = np.zeros((ns + 1, 2))
    e   = np.zeros(ns + 1)
    state = int(CtrlState.HIGH)

    for i in range(ns):
        e[i] = sp[i] - pv[i, 0]

        if state == CtrlState.HIGH and e[i] >= -ctrl.hys:
            op[i] = 100.0
        elif state == CtrlState.HIGH and e[i] < -ctrl.hys:
            op[i] = -100.0
            state = int(CtrlState.LOW)
        elif state == CtrlState.LOW and e[i] <= ctrl.hys:
            op[i] = -100.0
        elif state == CtrlState.LOW and e[i] > ctrl.hys:
            op[i] = 100.0
            state = int(CtrlState.HIGH)

        iop = max(0, i - ndelay)
        y = odeint(_process, pv[i], [0, Ts],
                   args=(op[iop], plant.V, plant.T, plant.xi))

        noise = (np.random.randint(3) - 1) / 10.0 if i % 10 == 0 else 0.0
        pv[i + 1, 0] = y[-1, 0] + noise
        pv[i + 1, 1] = y[-1, 1]

    op[ns] = op[ns - 1]
    return t, pv[:, 0], op, sp, e

# === 6. PLOTTING ===

def build_chart(t, pv, op, sp):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("Process Variable & Set Point", "Controller Output (OP)"),
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=t, y=sp, name="SP", mode="lines",
        line=dict(color=C_SP, dash="dash", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=pv, name="PV", mode="lines",
        line=dict(color=C_PV, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=op, name="OP", mode="lines",
        line=dict(color=C_OP, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(255,165,0,0.15)"), row=2, col=1)

    fig.update_xaxes(**_AX, title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(**_AX, row=1, col=1)
    fig.update_yaxes(**_AX, title_text="Level (%)", row=1, col=1)
    fig.update_yaxes(**_AX, title_text="OP (%)", row=2, col=1)
    fig.update_layout(**_BASE, height=560)
    return fig

# === 7. SIDEBAR ===

def render_sidebar():
    with st.sidebar:
        st.title("🪣 Level Controller")
        st.caption("Bang-Bang Controller with Hysteresis")
        st.divider()

        # --- Presets ---
        st.subheader("⚡ Presets")
        c1, c2, c3 = st.columns(3)
        if c1.button("⚡ Fast", use_container_width=True):
            for k, v in dict(V=6.0, T=40.0, xi=2.0, Tdelay=2.0, hys=0.0).items():
                st.session_state[k] = v
        if c2.button("🐢 Slow", use_container_width=True):
            for k, v in dict(V=2.0, T=150.0, xi=4.0, Tdelay=10.0, hys=1.0).items():
                st.session_state[k] = v
        if c3.button("↺ Reset", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
        st.divider()

        # --- Plant Parameters ---
        st.subheader("🏗️ Plant Parameters")
        st.slider("Process Gain V", 0.5, 10.0, key="V", step=0.5,
                  help="Static gain of the second-order system")
        st.slider("Time Constant T (s)", 10.0, 300.0, key="T", step=5.0,
                  help="Dominant time constant of the plant")
        st.slider("Damping ξ", 0.5, 5.0, key="xi", step=0.1,
                  help="Damping factor (>1 = overdamped)")
        st.slider("Dead Time Tdelay (s)", 0.0, 30.0, key="Tdelay", step=1.0,
                  help="Transport/dead time delay")
        st.divider()

        # --- Controller ---
        st.subheader("🕹️ Controller")
        st.slider("Hysteresis Band ±", 0.0, 10.0, key="hys", step=0.1,
                  help="Dead-band around set point (0 = ideal bang-bang)")
        st.divider()

        # --- Set Point Profile ---
        st.subheader("🎯 Set Point Profile")
        st.number_input("SP Phase 1 (%)", key="sp1", min_value=0.0, max_value=200.0, step=5.0)
        st.number_input("SP Phase 2 (%)", key="sp2", min_value=0.0, max_value=200.0, step=5.0)
        st.number_input("SP Phase 3 (%)", key="sp3", min_value=0.0, max_value=200.0, step=5.0)
        st.number_input("Step 1→2 at (s)", key="t1", min_value=10.0, max_value=900.0, step=50.0)
        st.number_input("Step 2→3 at (s)", key="t2", min_value=10.0, max_value=950.0, step=50.0)
        st.divider()

        # --- Simulation Settings ---
        st.subheader("⚙️ Simulation")
        st.number_input("End Time (s)", key="Tend", min_value=100.0, max_value=5000.0, step=100.0)
        st.divider()

        # --- Save / Load Config ---
        st.subheader("💾 Save / Load Config")
        cfg = {k: safe_get(k) for k in DEFAULTS}
        st.download_button(
            "⬇️ Download Config",
            data=json.dumps(cfg, indent=2),
            file_name="level_ctrl_config.json",
            mime="application/json",
            use_container_width=True,
        )

        def on_upload():
            try:
                raw = json.load(st.session_state["uploaded_cfg"])
                for k, lo, hi in [
                    ("V", 0.5, 10.0), ("T", 10.0, 300.0), ("xi", 0.5, 5.0),
                    ("Tdelay", 0.0, 30.0), ("hys", 0.0, 10.0),
                    ("sp1", 0.0, 200.0), ("sp2", 0.0, 200.0), ("sp3", 0.0, 200.0),
                    ("t1", 10.0, 900.0), ("t2", 10.0, 950.0), ("Tend", 100.0, 5000.0),
                ]:
                    if k in raw:
                        st.session_state[k] = clamp(float(raw[k]), lo, hi)
                st.toast("Configuration loaded ✓")
            except Exception:
                st.toast("⚠️ Invalid config file")

        st.file_uploader(
            "⬆️ Upload Config", type="json",
            key="uploaded_cfg", on_change=on_upload,
            label_visibility="collapsed",
        )

# === 8. TABS ===

def render_tab_simulation(t, pv, op, sp, e):
    # --- Metrics ---
    overshoot = float(np.max(pv) - sp[0]) if np.max(pv) > sp[0] else 0.0
    switches   = int(np.sum(np.abs(np.diff(op)) > 50))
    mean_err   = float(np.mean(np.abs(e)))

    m1, m2, m3 = st.columns(3)
    m1.metric("Max Overshoot", f"{overshoot:.1f} %",
              help="Peak PV above first set point")
    m2.metric("Controller Switches", f"{switches}",
              help="Number of bang-bang state transitions")
    m3.metric("Mean |Error|", f"{mean_err:.2f} %",
              help="Average absolute control error over simulation")

    # --- Chart ---
    fig = build_chart(t, pv, op, sp)
    st.plotly_chart(fig, use_container_width=True)

    # --- CSV Download ---
    df_csv = io.StringIO()
    df_csv.write("time,sp,pv,op\n")
    for i in range(len(t)):
        df_csv.write(f"{t[i]:.2f},{sp[i]:.4f},{pv[i]:.4f},{op[i]:.4f}\n")
    st.download_button(
        "⬇️ Download CSV",
        data=df_csv.getvalue(),
        file_name="level_ctrl_simulation.csv",
        mime="text/csv",
    )

def render_tab_theory():
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("Plant Model — Second-Order System")
        st.latex(r"T^2 \ddot{y}(t) + 2\,\xi\,T\,\dot{y}(t) + y(t) = V \cdot u(t - T_{delay})")
        st.markdown("""
| Symbol | Description |
|---|---|
| $V$ | Process gain |
| $T$ | Time constant (s) |
| $\\xi$ | Damping factor |
| $T_{delay}$ | Dead time (s) |
| $y(t)$ | Process variable (level) |
| $u(t)$ | Controller output |
""")
        st.divider()
        st.subheader("Bang-Bang Controller with Hysteresis")
        st.latex(r"""
u(t) = \begin{cases}
+100\% & \text{if } e(t) \geq -h \\
-100\% & \text{if } e(t) < -h
\end{cases}
\quad \text{(switching from HIGH)}
""")
        st.latex(r"""
u(t) = \begin{cases}
-100\% & \text{if } e(t) \leq +h \\
+100\% & \text{if } e(t) > +h
\end{cases}
\quad \text{(switching from LOW)}
""")
        st.markdown(r"""
where $e(t) = SP - PV$ is the control error and $h$ is the **hysteresis band**.

- **$h = 0$** → ideal bang-bang (maximum switching frequency)
- **$h > 0$** → fewer switches, reduced wear on actuators, larger oscillation
""")

    with col2:
        st.subheader("How It Works")
        st.info("""
**Bang-bang (two-position) control** is the simplest form of feedback control:
the output is either fully ON or fully OFF. It is widely used in:

- 🌡️ Thermostats (heating ON/OFF)
- 🪣 Tank level control (pump ON/OFF)
- ⚡ Battery chargers (charge ON/OFF)
""")
        st.subheader("Effect of Hysteresis")
        st.success("**Small hysteresis (h ≈ 0):** Very tight control, but the controller switches rapidly → high actuator wear.")
        st.warning("**Large hysteresis:** Fewer switches, longer actuator life, but larger oscillation amplitude around the set point.")

        st.subheader("Dead Time Effect")
        st.error("""
Dead time $T_{delay}$ makes bang-bang control more challenging:
the controller reacts to **old** process values, causing the system to overshoot
before the output change takes effect. Larger dead times → larger oscillations.
""")

def render_tab_exercises():
    st.subheader("Exercises — Bang-Bang Level Controller")

    with st.expander("📝 Exercise 1: Effect of Hysteresis", expanded=True):
        st.markdown("""
**Task:** Run the simulation with hysteresis $h = 0$ (ideal bang-bang),
then increase it to $h = 2.0$ and $h = 5.0$.

Observe how the number of controller switches and the oscillation amplitude change.

**Question:** What is the trade-off when choosing the hysteresis band?
""")
        with st.expander("🔍 Solution"):
            st.markdown("Larger $h$ → fewer switches, larger PV oscillation. Smaller $h$ → tighter control, more switches.")
        with st.expander("💡 Explanation"):
            st.markdown(r"""
The hysteresis band creates a **dead zone** around the set point.
The controller only switches state when the error $e$ exceeds $\pm h$.

- **Switches** $\approx \frac{1}{2 h \cdot T}$ — inversely proportional to $h$ and $T$
- **Amplitude** of oscillation $\approx 2 h$ (for slow plants)

In practice, $h$ is chosen to balance actuator lifetime vs. control precision.
""")

    with st.expander("📝 Exercise 2: Effect of Dead Time"):
        st.markdown(r"""
**Task:** Set $T_{delay} = 0$ s and note the oscillation.
Then increase it to $T_{delay} = 15$ s and $T_{delay} = 30$ s.

**Question:** Why does dead time worsen bang-bang control performance?
""")
        with st.expander("🔍 Solution"):
            st.markdown("Higher dead time → larger overshoot and oscillations, because the controller reacts to delayed information.")
        with st.expander("💡 Explanation"):
            st.markdown(r"""
Dead time means the controller only sees the effect of its action after $T_{delay}$ seconds.

During this delay the process continues moving **past** the set point, causing overshoot.
The controller then switches too late, amplifying the oscillation.

**Rule of thumb:** For bang-bang control, keep $T_{delay} \ll T$ to minimize overshoot.
""")

    with st.expander("📝 Exercise 3: Set Point Steps"):
        st.markdown("""
**Task:** Configure a three-step set point profile:
- Phase 1: SP = 50 % (0–400 s)
- Phase 2: SP = 120 % (400–700 s)
- Phase 3: SP = 80 % (700–1000 s)

Observe how the controller handles both upward and downward set point changes.
""")
        with st.expander("🔍 Solution"):
            st.markdown("The controller switches to HIGH for upward steps and to LOW for downward steps — both with similar overshoot characteristics.")
        with st.expander("💡 Explanation"):
            st.markdown("""
Bang-bang control is **symmetric**: it can drive the process up (OP = +100%) or down (OP = −100%).
For a downward step, the controller immediately switches to LOW output.

The response speed depends on $V/T$ — the steepness of the open-loop ramp.
""")

# === 9. MAIN ===

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    plant = PlantConfig(
        V=safe_get("V"), T=safe_get("T"),
        xi=safe_get("xi"), Tdelay=safe_get("Tdelay"),
    )
    ctrl = CtrlConfig(
        hys=safe_get("hys"),
        sp1=safe_get("sp1"), sp2=safe_get("sp2"), sp3=safe_get("sp3"),
        t1=safe_get("t1"), t2=safe_get("t2"),
        Tend=safe_get("Tend"), Ts=0.5,
    )

    with st.spinner("Simulating..."):
        t, pv, op, sp, e = simulate(plant, ctrl)

    tab1, tab2, tab3 = st.tabs([
        "🕹️ Simulation",
        "📘 Theory",
        "🧪 Exercises",
    ])

    with tab1:
        render_tab_simulation(t, pv, op, sp, e)
    with tab2:
        render_tab_theory()
    with tab3:
        render_tab_exercises()

if __name__ == "__main__":
    main()
