import streamlit as st
import numpy as np
from scipy import signal as scipy_signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ==========================================
# --- 0. PAGE CONFIG & STATE MANAGEMENT ---
# ==========================================
st.set_page_config(page_title="Oscillation & Damping", layout="wide", page_icon="🔔")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size: 18px !important; font-weight: bold !important; }
    .stMetric { background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #303030; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
T_FIXED  = 1.0   # Time constant – fixed to keep the focus on damping
T_END    = 50.0  # Simulation duration [s]
DT       = 0.05  # Time step [s]
STEP_AMP = 1.0   # Unit step amplitude [–]

# Validated option ranges (used both in sliders and JSON import)
XI_MIN, XI_MAX, XI_STEP = 0.0, 2.0,  0.05
V_MIN,  V_MAX,  V_STEP  = 0.5,  5.0,  0.1

DEFAULTS: dict = {
    "xi": 1.0,
    "V":  1.0,
}


def init_state() -> None:
    """Initialise every session-state key with its default value if not present."""
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def safe_get(key: str):
    """Return session-state value, falling back to the default if missing."""
    return st.session_state.get(key, DEFAULTS[key])


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ==========================================
# --- 1. CALLBACKS ---
# ==========================================
def on_upload_callback() -> None:
    """Load a JSON config and validate every value before applying."""
    uploader = st.session_state.get("json_uploader")
    if uploader is None:
        return
    try:
        data = json.load(uploader)
    except Exception as exc:
        st.error(f"Could not read the file – is it a valid JSON file? ({exc})")
        return

    mapping = {
        "xi": lambda v: clamp(float(v), XI_MIN, XI_MAX),
        "V":  lambda v: clamp(float(v), V_MIN,  V_MAX),
    }
    for key, validator in mapping.items():
        if key in data:
            try:
                st.session_state[key] = validator(data[key])
            except Exception:
                st.session_state[key] = DEFAULTS[key]

    st.toast("✅ Settings loaded!", icon="💾")


# ==========================================
# --- 2. SIMULATION ---
# ==========================================
# All arguments are primitives → no cache-hashing issues across Streamlit versions.

@st.cache_data(show_spinner=False)
def compute_response(xi: float, V: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the step response of a 2nd-order system.

    Transfer function: H(s) = V / (T²s² + 2ξTs + 1)
    with T = T_FIXED = 1.0, step amplitude = STEP_AMP.

    Returns (t, y, u) – time axis, output, and input arrays.
    """
    xi = max(xi, 0.001)   # guard: zero damping would be purely imaginary
    V  = max(V,  0.001)

    t = np.linspace(0.0, T_END, int(T_END / DT) + 1)
    u = np.full_like(t, STEP_AMP)

    num = [V]
    den = [T_FIXED**2, 2.0 * xi * T_FIXED, 1.0]

    sys_tf       = scipy_signal.TransferFunction(num, den)
    t_out, y_out, _ = scipy_signal.lsim(sys_tf, u, t)

    return t_out, y_out, u


def response_metrics(t: np.ndarray, y: np.ndarray, V: float) -> dict:
    """Compute key step-response quality metrics."""
    y_ss = float(V * STEP_AMP)   # theoretical steady-state value

    # Overshoot
    y_max    = float(np.max(y))
    overshoot = max(0.0, (y_max - y_ss) / y_ss * 100.0) if y_ss > 1e-9 else 0.0

    # Rise time: 10 % → 90 % of steady state
    idx_10 = np.where(y >= 0.1 * y_ss)[0]
    idx_90 = np.where(y >= 0.9 * y_ss)[0]
    t_rise = float(t[idx_90[0]] - t[idx_10[0]]) if (len(idx_10) and len(idx_90)) else None

    # Settling time: last moment the output leaves the ±2 % band
    band      = 0.02 * abs(y_ss)
    outside   = np.where(np.abs(y - y_ss) > band)[0]
    t_settle  = float(t[outside[-1]]) if len(outside) > 0 else 0.0

    return {
        "y_ss":      y_ss,
        "overshoot": overshoot,
        "t_rise":    t_rise,
        "t_settle":  t_settle,
    }


# ==========================================
# --- 3. REGIME HELPER ---
# ==========================================
def show_regime_badge(xi: float) -> None:
    """Display a coloured status box describing the current damping regime."""
    if xi < 0.95:
        st.warning(
            f"**ξ = {xi:.2f} → Underdamped** – "
            "the system overshoots and oscillates before settling down."
        )
    elif xi <= 1.05:
        st.success(
            f"**ξ = {xi:.2f} → Critically damped** – "
            "the fastest possible response without any overshoot."
        )
    else:
        st.info(
            f"**ξ = {xi:.2f} → Overdamped** – "
            "no oscillation, but the system is slower to reach its target."
        )


# ==========================================
# --- 4. SIDEBAR ---
# ==========================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("1. Set Parameters")

        st.slider(
            "Damping Factor ξ (xi)",
            min_value=XI_MIN, max_value=XI_MAX, step=XI_STEP,
            key="xi",
            help=(
                "ξ < 1: system oscillates (underdamped).  "
                "ξ = 1: fastest response without overshoot (critically damped).  "
                "ξ > 1: slow, no oscillation (overdamped)."
            ),
        )
        st.slider(
            "System Gain V",
            min_value=V_MIN, max_value=V_MAX, step=V_STEP,
            key="V",
            help="Scales the final output level. Steady-state output = V × step amplitude.",
        )

        st.divider()

        # Quick-preset buttons – on_click callbacks fire before the next render,
        # so they can safely update the slider's session-state key without conflict.
        st.markdown("**Quick presets:**")

        def _set_under():    st.session_state["xi"] = 0.3
        def _set_critical(): st.session_state["xi"] = 1.0
        def _set_over():     st.session_state["xi"] = 1.5

        col1, col2, col3 = st.columns(3)
        col1.button("Under-\ndamped",  use_container_width=True, help="Sets ξ = 0.3", on_click=_set_under)
        col2.button("Critical",        use_container_width=True, help="Sets ξ = 1.0", on_click=_set_critical)
        col3.button("Over-\ndamped",   use_container_width=True, help="Sets ξ = 1.5", on_click=_set_over)

        st.divider()

        # Save / load
        st.header("💾 Settings")
        conf_data = {k: safe_get(k) for k in DEFAULTS}
        st.download_button(
            label="📥 Save settings (JSON)",
            data=json.dumps(conf_data, indent=2),
            file_name="oscillation_config.json",
            mime="application/json",
            use_container_width=True,
        )
        st.file_uploader(
            "📤 Load settings (JSON)",
            type=["json"],
            key="json_uploader",
            on_change=on_upload_callback,
        )


# ==========================================
# --- 5. MAIN UI ---
# ==========================================
def main() -> None:
    init_state()

    st.title("🔔 Oscillation & Damping – How Systems React to a Push")
    st.markdown(
        "What happens when you suddenly push a system from standstill? "
        "Does it shoot past its target, swing back and forth, or creep up slowly? "
        "The answer depends on one number: the **damping factor ξ**."
    )

    render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "1. 🚀 Try It Out",
        "2. 📘 Explanation",
        "3. 📝 Exercises",
    ])

    # ── TAB 1: Simulation ─────────────────────────────────────────────────
    with tab1:
        xi = float(safe_get("xi"))
        V  = float(safe_get("V"))

        t, y, u = compute_response(xi, V)
        m       = response_metrics(t, y, V)

        # --- Regime indicator ---
        show_regime_badge(xi)
        st.markdown("---")

        # --- 3 metrics ---
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Steady-State Output",
            f"{m['y_ss']:.2f}",
            help="The value the output settles at (= System Gain V × step size).",
        )
        c2.metric(
            "Overshoot",
            f"{m['overshoot']:.1f} %",
            help="How far the output exceeds its final value before settling. "
                 "0 % means no overshoot.",
        )
        c3.metric(
            "Settling Time",
            f"{m['t_settle']:.1f} s",
            help="Time until the output stays permanently within ±2 % of its final value.",
        )

        st.markdown("---")

        # --- Dual subplot: output on top, input on bottom ---
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("System Output y(t)", "Input – Unit Step u(t)"),
            vertical_spacing=0.12,
        )

        # Output trace
        fig.add_trace(
            go.Scatter(x=t, y=y, mode="lines", name="Output y(t)",
                       line=dict(color="#FF9F1C", width=3)),
            row=1, col=1,
        )
        # Steady-state reference line
        fig.add_hline(
            y=m["y_ss"], line_dash="dot", line_color="#888888", line_width=1.5,
            annotation_text=f"Steady state = {m['y_ss']:.2f}",
            annotation_position="bottom right",
            row=1, col=1,
        )
        # ±2 % settling band
        band = 0.02 * m["y_ss"]
        fig.add_hrect(
            y0=m["y_ss"] - band, y1=m["y_ss"] + band,
            fillcolor="#00CC96", opacity=0.08, line_width=0,
            row=1, col=1,
        )

        # Input trace
        fig.add_trace(
            go.Scatter(x=t, y=u, mode="lines", name="Input u(t)",
                       line=dict(color="#636EFA", width=2)),
            row=2, col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=580,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(
            title_text="Time (s)", row=2, col=1,
            showgrid=True, gridcolor="#333333",
            zeroline=True, zerolinecolor="#888888",
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="#333333",
            zeroline=True, zerolinecolor="#888888",
            row=1, col=1,
        )
        fig.update_yaxes(
            title_text="Amplitude", row=1, col=1,
            showgrid=True, gridcolor="#333333",
            zeroline=True, zerolinecolor="#888888",
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#333333",
            zeroline=True, zerolinecolor="#888888",
            row=2, col=1,
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- CSV download ---
        csv_rows = ["Time [s],Output y(t),Input u(t)"]
        csv_rows += [f"{ti:.3f},{yi:.6f},{ui:.3f}" for ti, yi, ui in zip(t, y, u)]
        csv = "\n".join(csv_rows).encode("utf-8")
        _, col_btn = st.columns([3, 1])
        with col_btn:
            st.download_button(
                label="📊 Download data (CSV)",
                data=csv,
                file_name="oscillation_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── TAB 2: Explanation ─────────────────────────────────────────────────
    with tab2:
        st.header("📘 What Is Damping – and Why Does It Matter?")

        st.markdown("""
> **Key idea:** When you push a real-world system to a new target, it rarely
> jumps there instantly. How it gets there – overshoot, oscillation, or slow creep –
> is controlled by the **damping factor ξ (xi)**.
        """)

        st.divider()

        st.subheader("🚗 The Car Suspension Analogy")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
Imagine driving over a speed bump.

- The **spring** pushes the wheel back down – that is the *restoring force*.
- The **shock absorber** (damper) slows down the bounce – that is the *damping*.

Without any shock absorber, the car would keep bouncing up and down forever.
With just the right amount of damping, it settles smoothly after one dip.
With too much damping, it sinks back very slowly, like moving through honey.

**ξ is the number that describes how strong the shock absorber is.**
            """)
        with col_b:
            st.markdown("""
The same principle appears everywhere:

| System | Spring force | Damper |
|---|---|---|
| Car suspension | Spring | Shock absorber |
| Door closer | Return spring | Oil brake |
| Electronic circuit | Capacitor | Resistor |
| Robotic arm | Motor torque | Friction / control |

In all cases, ξ controls whether the system oscillates, settles cleanly,
or creeps slowly to its target.
            """)

        st.divider()

        st.subheader("The Three Regimes")
        col_u, col_c, col_o = st.columns(3)

        with col_u:
            st.warning("**⚠️ Underdamped  (ξ < 1)**")
            st.markdown("""
The system **overshoots** its target and swings back and forth before settling.

- Responds quickly at first
- Oscillations die out gradually
- The smaller ξ, the more cycles before settling

**Analogy:** A car with worn-out shock absorbers – it keeps bouncing after a bump.

*Typical use: when speed matters more than precision.*
            """)

        with col_c:
            st.success("**✅ Critically damped  (ξ = 1)**")
            st.markdown("""
The system reaches its target **as fast as possible without any overshoot**.

- No oscillation at all
- Fastest approach without crossing the target
- The theoretical "best of both worlds"

**Analogy:** A well-tuned car suspension – settles after the bump in one smooth motion.

*Often the design target in control systems.*
            """)

        with col_o:
            st.info("**🐢 Overdamped  (ξ > 1)**")
            st.markdown("""
The system approaches its target **slowly**, like moving through thick liquid.

- No overshoot, no oscillation
- The higher ξ, the slower the response
- Very safe, but sluggish

**Analogy:** A car with extremely stiff shock absorbers – barely any bounce, but also barely any movement.

*Used where overshoot is absolutely not allowed, even at the cost of speed.*
            """)

        st.divider()

        st.subheader("🎯 The Sweet Spot: ξ ≈ 0.7")
        st.markdown("""
In practice, a damping factor around **ξ = 0.7** is often a good compromise:

- The overshoot is small (about 5 %)
- The response is fast
- The system settles quickly

This is why you will find ξ ≈ 0.7 in many real engineering designs –
from the steering of a car to the position control of a hard drive read head.

The green band in the plot shows the ±2 % settling zone.
Once the output stays inside that band, the system is considered settled.
        """)

    # ── TAB 3: Exercises ───────────────────────────────────────────────────
    with tab3:
        st.header("📝 Exercises")
        st.markdown(
            "Use the **Damping Factor ξ slider** in the left sidebar "
            "– or the quick-preset buttons – and watch how the response changes."
        )

        with st.expander("⚡ Exercise 1 – Feel the Oscillation", expanded=True):
            st.markdown("""
**Task:** Set the system to strongly underdamped and count the oscillations.

**Settings:** ξ = **0.2**  (or press the *Underdamped* button)

**What you see:**
- The output shoots well past the steady-state line
- It swings back below the target, then above again – multiple times
- Check the *Overshoot* metric – it should be around **50 %**
- The *Settling Time* is very long

**Try lowering ξ to 0.1** – even more oscillations, even longer to settle.

→ *Very low damping = fast start, but the system cannot make up its mind.*
            """)

        with st.expander("✅ Exercise 2 – Find Critical Damping"):
            st.markdown("""
**Task:** Set the system to critical damping and confirm there is zero overshoot.

**Settings:** ξ = **1.0**  (or press the *Critical* button)

**What you see:**
- The output rises smoothly and touches the steady-state line exactly once – no bounce
- *Overshoot* metric reads **0 %**
- The *Settling Time* is shorter than the overdamped case

**Compare with ξ = 0.9** and **ξ = 1.1** – how close to 1.0 do you have to be
before the overshoot appears or the response becomes noticeably slower?

→ *Critical damping = the fastest you can go without any overshoot.*
            """)

        with st.expander("🐢 Exercise 3 – Overdamping"):
            st.markdown("""
**Task:** Observe how too much damping slows the system down.

**Settings:** ξ = **1.5**  (or press the *Overdamped* button)

**What you see:**
- No oscillation, no overshoot
- But the output rises very slowly – it may not even reach the target within the plot window

**Try ξ = 2.0** – even slower.

→ *Too much damping is safe, but sometimes frustratingly slow.*
            """)

        with st.expander("🎯 Exercise 4 – Find the Engineering Sweet Spot"):
            st.markdown("""
**Task:** Find a damping factor that gives fast response with only a tiny overshoot.

**Goal:** Overshoot < 10 %, Settling Time < 15 s

**Start with ξ = 0.7** and observe:
- Overshoot is around 5 %
- The output settles quickly

Now try **ξ = 0.5** and **ξ = 0.9** – compare speed and overshoot.

| ξ | Overshoot | Comment |
|---|---|---|
| 0.5 | ~16 % | Fast but bouncy |
| 0.7 | ~5 % | Good balance ✅ |
| 0.9 | ~1 % | Almost no overshoot, slightly slower |

→ *ξ ≈ 0.7 is a classic engineering rule of thumb for a well-behaved system.*
            """)

        with st.expander("📈 Exercise 5 – Effect of System Gain V"):
            st.markdown("""
**Task:** Change the System Gain V and observe what it affects – and what it does NOT.

**Settings:** ξ = **0.5** (keep fixed throughout this exercise)

**Step 1:** V = **1.0** → note the Steady-State Output and Overshoot %

**Step 2:** V = **2.0** → the output now settles at double the height

**Step 3:** V = **3.0** → triple the height

**Key observation:** The *Overshoot %* and *Settling Time* do NOT change.
Only the final output level changes.

→ *System Gain V controls how much the output amplifies the input.
   It has no effect on the oscillation behaviour – that is entirely governed by ξ.*
            """)


if __name__ == "__main__":
    main()
