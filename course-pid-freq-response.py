import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Frequency Response & Bode Plots", layout="wide", page_icon="📡")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size:16px !important; font-weight:bold !important; }
    .stMetric { background:#0E1117; padding:12px; border-radius:8px; border:1px solid #303030; }
    section[data-testid="stSidebar"] { min-width: 310px; max-width: 310px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SYS_LABELS = [
    "1st Order Low-Pass",
    "1st Order High-Pass",
    "2nd Order Low-Pass",
    "2nd Order High-Pass",
]

FC_MIN,  FC_MAX,  FC_STEP  = 0.01, 50.0, 0.01
FN_MIN,  FN_MAX,  FN_STEP  = 0.1,  50.0, 0.1
XI_MIN,  XI_MAX,  XI_STEP  = 0.05,  2.0, 0.05
V_MIN,   V_MAX,   V_STEP   = 0.1,   5.0, 0.1
FT_MIN,  FT_MAX,  FT_STEP  = 0.01, 100.0, 0.01

DEFAULTS: dict = {
    "sys_label": "1st Order Low-Pass",
    "f_c":       1.0,
    "f_n":       5.0,
    "xi":        0.5,
    "V":         1.0,
    "f_test_hz": 0.5,
}

DARK = "#0D1117"
GRID = "#21262D"
TEXT = "#C9D1D9"
ACC1 = "#58A6FF"   # blue  – input signal
ACC2 = "#3FB950"   # green – output signal
ACC3 = "#F78166"   # orange – test freq marker


# ─── STATE HELPERS ────────────────────────────────────────────────────────────
def init_state() -> None:
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def safe_get(key: str):
    return st.session_state.get(key, DEFAULTS[key])


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ─── TRANSFER FUNCTION MATH ───────────────────────────────────────────────────
def freq_response(sys_label: str, f_arr: np.ndarray,
                  f_c: float, f_n: float, xi: float, V: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (gain_db, phase_deg) arrays for given frequency array."""
    w   = 2.0 * np.pi * f_arr
    wc  = 2.0 * np.pi * f_c
    wn  = 2.0 * np.pi * f_n

    if sys_label == "1st Order Low-Pass":
        H = 1.0 / (1.0 + 1j * w / wc)
    elif sys_label == "1st Order High-Pass":
        H = (1j * w / wc) / (1.0 + 1j * w / wc)
    elif sys_label == "2nd Order Low-Pass":
        denom = wn**2 - w**2 + 2j * xi * wn * w
        H = V * wn**2 / denom
    else:  # 2nd Order High-Pass
        denom = wn**2 - w**2 + 2j * xi * wn * w
        H = -V * w**2 / denom

    mag   = np.abs(H)
    gain  = 20.0 * np.log10(np.maximum(mag, 1e-12))
    phase = np.degrees(np.angle(H))
    return gain, phase


def single_freq_response(sys_label: str, f_test: float,
                          f_c: float, f_n: float, xi: float, V: float
                          ) -> tuple[float, float]:
    """Return (gain_linear, phase_deg) at a single test frequency."""
    g_db, ph = freq_response(sys_label, np.array([f_test]), f_c, f_n, xi, V)
    return float(10.0 ** (g_db[0] / 20.0)), float(ph[0])


# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────
_LAYOUT_BASE = dict(
    paper_bgcolor=DARK, plot_bgcolor=DARK,
    font=dict(color=TEXT, size=13),
    margin=dict(l=60, r=30, t=40, b=50),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)


def _apply_base(fig: go.Figure, **extra) -> go.Figure:
    fig.update_layout(**_LAYOUT_BASE, **extra)
    return fig


def make_time_plot(sys_label: str, f_test: float,
                   f_c: float, f_n: float, xi: float, V: float) -> go.Figure:
    """Full-width time domain: input sine + output sine."""
    gain_lin, phase_deg = single_freq_response(sys_label, f_test, f_c, f_n, xi, V)
    phase_rad = np.radians(phase_deg)

    # show at least 3 periods, capped at sensible range
    T_test = 1.0 / max(f_test, 0.001)
    t_end  = max(3.0 * T_test, 0.5)
    t_end  = min(t_end, 20.0)
    t = np.linspace(0.0, t_end, 2000)

    u_in  = np.sin(2.0 * np.pi * f_test * t)
    u_out = gain_lin * np.sin(2.0 * np.pi * f_test * t + phase_rad)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=u_in,  mode="lines",
                             name="Input u(t)",  line=dict(color=ACC1, width=2)))
    fig.add_trace(go.Scatter(x=t, y=u_out, mode="lines",
                             name="Output y(t)", line=dict(color=ACC2, width=2)))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Input & Output Signal", font=dict(size=15)),
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", y=1.12, x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=280,
    )
    return fig


def make_bode_plots(sys_label: str, f_test: float,
                    f_c: float, f_n: float, xi: float, V: float
                    ) -> tuple[go.Figure, go.Figure]:
    """Return (mag_fig, phase_fig) – full-width Bode plots."""
    f_lo = max(f_c if "1st" in sys_label else f_n, 0.05) / 100.0
    f_lo = max(f_lo, 0.001)
    f_hi = (f_c if "1st" in sys_label else f_n) * 200.0
    f_hi = min(f_hi, 1e5)
    f_arr = np.logspace(np.log10(f_lo), np.log10(f_hi), 800)

    gain_db, phase_deg = freq_response(sys_label, f_arr, f_c, f_n, xi, V)

    # test freq values
    g_test, ph_test = freq_response(sys_label, np.array([f_test]), f_c, f_n, xi, V)
    g_test  = float(g_test[0])
    ph_test = float(phase_deg[np.argmin(np.abs(f_arr - f_test))])

    # char freq label
    if "1st" in sys_label:
        f_char      = f_c
        char_label  = f"f_c = {f_c:.2f} Hz"
    else:
        f_char      = f_n
        char_label  = f"f_n = {f_n:.2f} Hz"

    def _vline_shapes(y0, y1):
        return [
            dict(type="line", x0=f_test, x1=f_test, y0=y0, y1=y1,
                 xref="x", yref="y",
                 line=dict(color=ACC3, width=1.5, dash="dot")),
            dict(type="line", x0=f_char, x1=f_char, y0=y0, y1=y1,
                 xref="x", yref="y",
                 line=dict(color="#E3B341", width=1.5, dash="dash")),
        ]

    def _vline_annots(y_test, y_char, y_max):
        return [
            dict(x=np.log10(f_test), y=y_test,
                 xref="x", yref="y", text=f"f={f_test:.2f} Hz",
                 showarrow=True, arrowhead=2, arrowcolor=ACC3,
                 font=dict(color=ACC3, size=11), ax=40, ay=-30),
            dict(x=np.log10(f_char), y=y_max * 0.92,
                 xref="x", yref="y", text=char_label,
                 showarrow=False, font=dict(color="#E3B341", size=11),
                 xanchor="left"),
        ]

    # ── magnitude ──────────────────────────────────────────────────────────────
    g_min = float(np.min(gain_db)) - 5
    g_max = float(np.max(gain_db)) + 5

    mag_fig = go.Figure()
    mag_fig.add_trace(go.Scatter(
        x=f_arr, y=gain_db, mode="lines",
        name="Gain [dB]", line=dict(color=ACC1, width=2.5)))
    mag_fig.add_trace(go.Scatter(
        x=[f_test], y=[g_test], mode="markers",
        name="Test freq", marker=dict(color=ACC3, size=10, symbol="circle")))
    mag_fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Bode Plot – Magnitude", font=dict(size=15)),
        xaxis=dict(type="log", title="Frequency [Hz]", gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title="Gain [dB]", gridcolor=GRID, zerolinecolor=GRID),
        shapes=_vline_shapes(g_min, g_max),
        annotations=_vline_annots(g_test, g_max, g_max),
        legend=dict(orientation="h", y=1.12, x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=300,
    )

    # ── phase ──────────────────────────────────────────────────────────────────
    ph_min = float(np.min(phase_deg)) - 10
    ph_max = float(np.max(phase_deg)) + 10

    ph_fig = go.Figure()
    ph_fig.add_trace(go.Scatter(
        x=f_arr, y=phase_deg, mode="lines",
        name="Phase [°]", line=dict(color=ACC2, width=2.5)))
    ph_fig.add_trace(go.Scatter(
        x=[f_test], y=[ph_test], mode="markers",
        name="Test freq", marker=dict(color=ACC3, size=10, symbol="circle")))
    ph_fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Bode Plot – Phase", font=dict(size=15)),
        xaxis=dict(type="log", title="Frequency [Hz]", gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title="Phase [°]", gridcolor=GRID, zerolinecolor=GRID),
        shapes=_vline_shapes(ph_min, ph_max),
        annotations=_vline_annots(ph_test, ph_max, ph_max),
        legend=dict(orientation="h", y=1.12, x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=300,
    )

    return mag_fig, ph_fig


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ System Parameters")

        sys_choice = st.selectbox(
            "System Type",
            SYS_LABELS,
            index=SYS_LABELS.index(str(safe_get("sys_label"))),
        )
        st.session_state["sys_label"] = sys_choice

        is_2nd = "2nd" in sys_choice
        st.divider()

        if not is_2nd:
            st.subheader("1st Order Parameters")
            fc = st.slider("Cutoff Frequency f_c [Hz]",
                           FC_MIN, FC_MAX, float(safe_get("f_c")), FC_STEP)
            st.session_state["f_c"] = fc
        else:
            st.subheader("2nd Order Parameters")
            fn = st.slider("Natural Frequency f_n [Hz]",
                           FN_MIN, FN_MAX, float(safe_get("f_n")), FN_STEP)
            xi = st.slider("Damping Ratio ξ",
                           XI_MIN, XI_MAX, float(safe_get("xi")), XI_STEP)
            V  = st.slider("DC Gain V",
                           V_MIN, V_MAX, float(safe_get("V")), V_STEP)
            st.session_state["f_n"] = fn
            st.session_state["xi"]  = xi
            st.session_state["V"]   = V

        st.divider()
        st.subheader("Test Frequency")
        f_test = st.slider("f_test [Hz]",
                           FT_MIN, FT_MAX, float(safe_get("f_test_hz")), FT_STEP)
        st.session_state["f_test_hz"] = f_test

        st.divider()
        # ── save / load ────────────────────────────────────────────────────────
        st.subheader("💾 Save / Load")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save JSON", use_container_width=True):
                data = {k: safe_get(k) for k in DEFAULTS}
                st.download_button("⬇ Download",
                                   data=json.dumps(data, indent=2),
                                   file_name="freq_response_params.json",
                                   mime="application/json",
                                   use_container_width=True)
        with col2:
            upload = st.file_uploader("Load JSON", type="json",
                                       label_visibility="collapsed")
            if upload is not None:
                _on_upload(upload)


def _on_upload(upload) -> None:
    try:
        raw = json.loads(upload.read().decode())
        for k, v in raw.items():
            if k in DEFAULTS:
                st.session_state[k] = type(DEFAULTS[k])(v)
        st.success("Parameters loaded.")
    except Exception as e:
        st.error(f"Invalid file: {e}")


# ─── TABS ─────────────────────────────────────────────────────────────────────
def tab_single_freq() -> None:
    sys_label = str(safe_get("sys_label"))
    f_c       = float(safe_get("f_c"))
    f_n       = float(safe_get("f_n"))
    xi        = float(safe_get("xi"))
    V         = float(safe_get("V"))
    f_test    = float(safe_get("f_test_hz"))
    is_2nd    = "2nd" in sys_label

    gain_lin, phase_deg = single_freq_response(sys_label, f_test, f_c, f_n, xi, V)
    gain_db  = 20.0 * np.log10(max(gain_lin, 1e-12))

    # ── info box ──────────────────────────────────────────────────────────────
    if is_2nd:
        f_char = f_n
        char_str = f"f_n = {f_n:.2f} Hz, ξ = {xi:.2f}, V = {V:.2f}"
        ratio = f_test / f_n
    else:
        f_char = f_c
        char_str = f"f_c = {f_c:.2f} Hz"
        ratio = f_test / f_c

    if ratio < 0.1:
        regime = "well below the characteristic frequency – signal passes almost unchanged"
    elif ratio < 0.7:
        regime = "in the transition band – moderate attenuation / amplification"
    elif ratio < 1.4:
        regime = "near the characteristic frequency – strong filter effect"
    else:
        regime = "well above the characteristic frequency – signal strongly attenuated / amplified"

    st.info(
        f"**{sys_label}** | {char_str}\n\n"
        f"Test frequency **{f_test:.3f} Hz** is {regime}.\n"
        f"The output amplitude is **{gain_lin:.3f}×** the input "
        f"(**{gain_db:+.1f} dB**) with a phase shift of **{phase_deg:.1f}°**."
    )

    # ── metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Gain [dB]",          f"{gain_db:+.2f} dB")
    c2.metric("Gain [linear ratio]", f"{gain_lin:.4f}")
    c3.metric("Phase Shift",         f"{phase_deg:.1f}°")

    st.divider()

    # ── time domain ───────────────────────────────────────────────────────────
    time_fig = make_time_plot(sys_label, f_test, f_c, f_n, xi, V)
    st.plotly_chart(time_fig, use_container_width=True)

    # ── bode plots ────────────────────────────────────────────────────────────
    mag_fig, ph_fig = make_bode_plots(sys_label, f_test, f_c, f_n, xi, V)
    st.plotly_chart(mag_fig, use_container_width=True)
    st.plotly_chart(ph_fig,  use_container_width=True)


def tab_explanation() -> None:
    st.header("📖 Frequency Response & Bode Plots")

    st.markdown("""
### What is Frequency Response?

A linear system responds to a sinusoidal input with a sinusoidal output of the **same frequency**.
Only two things change: the **amplitude** (gain) and the **timing** (phase shift).

The **frequency response** describes how gain and phase vary over all frequencies.
A **Bode plot** visualises this on a logarithmic frequency axis — giving you the full picture at a glance.

---

### The Four System Types

| System | What it does | Characteristic parameter |
|---|---|---|
| **1st Order Low-Pass** | Passes low frequencies, blocks high ones | Cutoff frequency f_c |
| **1st Order High-Pass** | Passes high frequencies, blocks low ones | Cutoff frequency f_c |
| **2nd Order Low-Pass** | Steeper roll-off, possible resonance peak | Natural freq f_n, damping ξ |
| **2nd Order High-Pass** | High-freq emphasis, possible resonance peak | Natural freq f_n, damping ξ |

---

### Transfer Functions H(jω)

Let ω = 2π·f and ωc = 2π·f_c (or ωn = 2π·f_n):

**1st Order Low-Pass**
$$H(j\\omega) = \\frac{1}{1 + j\\,\\omega/\\omega_c}$$

At f = f_c: gain = 1/√2 ≈ **−3 dB**, phase = **−45°**

**1st Order High-Pass**
$$H(j\\omega) = \\frac{j\\,\\omega/\\omega_c}{1 + j\\,\\omega/\\omega_c}$$

At f = f_c: gain = 1/√2 ≈ **−3 dB**, phase = **+45°**

**2nd Order Low-Pass**
$$H(j\\omega) = \\frac{V \\cdot \\omega_n^2}{\\omega_n^2 - \\omega^2 + 2j\\,\\xi\\,\\omega_n\\,\\omega}$$

For ξ < 1/√2 ≈ 0.707 a **resonance peak** appears near f_n.

**2nd Order High-Pass**
$$H(j\\omega) = \\frac{-V \\cdot \\omega^2}{\\omega_n^2 - \\omega^2 + 2j\\,\\xi\\,\\omega_n\\,\\omega}$$

Same denominator – resonance near f_n for small ξ.

---

### Reading a Bode Plot

- **Magnitude plot** (top): gain in dB vs. log frequency.
  0 dB = unity gain, −20 dB = output is 10× smaller than input.
- **Phase plot** (bottom): phase shift in degrees vs. log frequency.
- **Orange dot & dashed line**: your current test frequency.
- **Yellow dashed line**: characteristic frequency (f_c or f_n).

---

### The −3 dB Point (Cutoff Frequency)

For a 1st-order filter, the cutoff frequency f_c is where:
- Gain drops to 1/√2 ≈ 0.707 of the DC gain → **−3 dB**
- Phase shift is exactly **±45°**

For a 2nd-order system, f_n is the **natural frequency** (undamped resonance).
The actual −3 dB point depends on the damping ratio ξ.

---

### Damping Ratio ξ (2nd Order Systems)

| ξ value | Behaviour |
|---|---|
| ξ < 0.5 | Strong resonance peak, oscillatory step response |
| ξ ≈ 0.707 | No peak (Butterworth), flat passband edge |
| ξ = 1.0 | Critically damped, no overshoot |
| ξ > 1.0 | Over-damped, two real poles |
""")


def tab_exercises() -> None:
    st.header("✏️ Exercises")

    st.markdown("""
Work through these exercises using the sidebar controls and the **Single Frequency** tab.

---

#### Exercise 1 – The −3 dB Point (1st Order)

1. Select **1st Order Low-Pass** and set f_c = 2 Hz.
2. Set the test frequency to f_test = 2 Hz.
3. Read off the three metrics.

**Questions:**
- What is the gain in dB? What is the gain as a linear ratio?
- What is the phase shift?
- Move f_test to 0.2 Hz and then to 20 Hz. How do the metrics change?

---

#### Exercise 2 – High-Pass vs. Low-Pass

1. Switch between **1st Order Low-Pass** and **1st Order High-Pass** (keep f_c = 2 Hz, f_test = 2 Hz).
2. Compare the gain and phase values.

**Questions:**
- At f_test = f_c, is the gain the same for both types?
- How does the phase sign differ?
- At very low and very high frequencies, which filter lets signals through?

---

#### Exercise 3 – Resonance in 2nd Order Systems

1. Select **2nd Order Low-Pass**, set f_n = 5 Hz, ξ = 0.1.
2. Slowly increase f_test from 1 Hz towards 5 Hz.

**Questions:**
- At what frequency is the gain maximum?
- What happens to the gain in dB? Is it above 0 dB?
- Now increase ξ to 1.0. Does the resonance peak disappear?

---

#### Exercise 4 – Butterworth Condition

1. Select **2nd Order Low-Pass**, f_n = 5 Hz, ξ = 0.707.
2. Set f_test = f_n.

**Questions:**
- Is there a resonance peak in the Bode magnitude plot?
- What is the phase at f_n?
- Compare with ξ = 0.3 and ξ = 1.5.

---

#### Exercise 5 – 2nd Order High-Pass

1. Select **2nd Order High-Pass**, f_n = 5 Hz, ξ = 0.3.
2. Sweep f_test from 0.5 Hz to 50 Hz.

**Questions:**
- Where does the gain peak occur?
- Below f_n, how fast does the gain roll off (roughly how many dB per decade)?
- Set V = 2.0. How does the Bode magnitude plot shift vertically?

---

#### Exercise 6 – Predicting the Output

1. Select **1st Order Low-Pass**, f_c = 1 Hz, f_test = 3 Hz.
2. Note the gain (linear) and phase shift from the metrics.
3. The input is: u(t) = sin(2π · 3 · t)

**Task:** Write down the expected output y(t) as a formula using the values you read off.
Verify it against the time-domain plot.
""")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    render_sidebar()

    st.title("📡 Frequency Response & Bode Plots")
    st.caption("Explore how filters shape signals across frequencies.")

    tabs = st.tabs(["📊 Single Frequency", "📖 Explanation", "✏️ Exercises"])
    with tabs[0]:
        tab_single_freq()
    with tabs[1]:
        tab_explanation()
    with tabs[2]:
        tab_exercises()


if __name__ == "__main__":
    main()
