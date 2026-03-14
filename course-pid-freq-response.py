import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

FC_MIN, FC_MAX, FC_STEP = 0.01, 50.0, 0.01
FN_MIN, FN_MAX, FN_STEP = 0.1,  50.0, 0.1
FT_MIN, FT_MAX, FT_STEP = 0.01, 100.0, 0.01

# Fixed internal parameters for 2nd order systems (not exposed to the user)
XI_FIXED = 0.5   # moderate damping – clear roll-off, no resonance peak
V_FIXED  = 1.0   # unity DC gain

DEFAULTS: dict = {
    "sys_label": "1st Order Low-Pass",
    "f_c":       1.0,
    "f_n":       5.0,
    "f_test_hz": 0.5,
}

DARK = "#0D1117"
GRID = "#21262D"
TEXT = "#C9D1D9"
ACC1 = "#58A6FF"   # blue   – input / magnitude curve
ACC2 = "#3FB950"   # green  – output / phase curve
ACC3 = "#F78166"   # orange – test-freq marker
YELL = "#E3B341"   # yellow – characteristic-freq line


# ─── STATE HELPERS ────────────────────────────────────────────────────────────
def init_state() -> None:
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def safe_get(key: str):
    return st.session_state.get(key, DEFAULTS[key])


# ─── SHARED LAYOUT ────────────────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor=DARK,
    plot_bgcolor=DARK,
    font=dict(color=TEXT, size=13),
    margin=dict(l=60, r=30, t=50, b=50),
)

_AX = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)


# ─── TRANSFER FUNCTION MATH ───────────────────────────────────────────────────
def freq_response(sys_label: str, f_arr: np.ndarray,
                  f_c: float, f_n: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (gain_db, phase_deg) for the given frequency array."""
    w  = 2.0 * np.pi * f_arr
    wc = 2.0 * np.pi * f_c
    wn = 2.0 * np.pi * f_n

    if sys_label == "1st Order Low-Pass":
        H = 1.0 / (1.0 + 1j * w / wc)
    elif sys_label == "1st Order High-Pass":
        H = (1j * w / wc) / (1.0 + 1j * w / wc)
    elif sys_label == "2nd Order Low-Pass":
        denom = wn**2 - w**2 + 2j * XI_FIXED * wn * w
        H = V_FIXED * wn**2 / denom
    else:  # 2nd Order High-Pass
        denom = wn**2 - w**2 + 2j * XI_FIXED * wn * w
        H = -V_FIXED * w**2 / denom

    mag   = np.abs(H)
    gain  = 20.0 * np.log10(np.maximum(mag, 1e-12))
    phase = np.degrees(np.angle(H))
    return gain, phase


def point_response(sys_label: str, f: float,
                   f_c: float, f_n: float) -> tuple[float, float]:
    """Return (gain_linear, phase_deg) at a single frequency."""
    g_db, ph = freq_response(sys_label, np.array([f]), f_c, f_n)
    return float(10.0 ** (g_db[0] / 20.0)), float(ph[0])


# ─── PLOTS ────────────────────────────────────────────────────────────────────
def make_time_plot(sys_label: str, f_test: float,
                   f_c: float, f_n: float) -> go.Figure:
    gain_lin, phase_deg = point_response(sys_label, f_test, f_c, f_n)
    phase_rad = np.radians(phase_deg)

    T_test = 1.0 / max(f_test, 0.001)
    t_end  = min(max(3.0 * T_test, 0.5), 20.0)
    t = np.linspace(0.0, t_end, 2000)

    u_in  = np.sin(2.0 * np.pi * f_test * t)
    u_out = gain_lin * np.sin(2.0 * np.pi * f_test * t + phase_rad)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=u_in,  mode="lines",
                             name="Input u(t)",  line=dict(color=ACC1, width=2)))
    fig.add_trace(go.Scatter(x=t, y=u_out, mode="lines",
                             name="Output y(t)", line=dict(color=ACC2, width=2)))
    fig.update_layout(
        **_BASE,
        title=dict(text="Input & Output Signal", font=dict(size=15)),
        xaxis=dict(title="Time [s]", **_AX),
        yaxis=dict(title="Amplitude", **_AX),
        legend=dict(orientation="h", y=1.12, x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=280,
    )
    return fig


def make_bode_magnitude(sys_label: str, f_test: float,
                        f_c: float, f_n: float,
                        f_char: float, char_label: str) -> go.Figure:
    f_lo  = max(f_char / 200.0, 0.001)
    f_hi  = min(f_char * 200.0, 1e5)
    f_arr = np.logspace(np.log10(f_lo), np.log10(f_hi), 800)

    gain_db, _ = freq_response(sys_label, f_arr, f_c, f_n)
    g_test, _  = freq_response(sys_label, np.array([f_test]), f_c, f_n)
    g_test = float(g_test[0])

    g_min = float(np.min(gain_db)) - 5
    g_max = float(np.max(gain_db)) + 5
    y_lbl = g_max - 0.08 * (g_max - g_min)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_arr, y=gain_db, mode="lines",
                             name="Gain [dB]", line=dict(color=ACC1, width=2.5)))
    fig.add_trace(go.Scatter(x=[f_test], y=[g_test], mode="markers",
                             name=f"f_test = {f_test:.2f} Hz",
                             marker=dict(color=ACC3, size=11, symbol="circle")))
    fig.update_layout(
        **_BASE,
        title=dict(text="Bode Plot – Magnitude", font=dict(size=15)),
        xaxis=dict(type="log", title="Frequency [Hz]", **_AX),
        yaxis=dict(title="Gain [dB]", range=[g_min, g_max], **_AX),
        shapes=[
            dict(type="line", x0=f_test, x1=f_test, y0=g_min, y1=g_max,
                 xref="x", yref="y", line=dict(color=ACC3, width=1.5, dash="dot")),
            dict(type="line", x0=f_char, x1=f_char, y0=g_min, y1=g_max,
                 xref="x", yref="y", line=dict(color=YELL, width=1.5, dash="dash")),
        ],
        annotations=[
            dict(x=f_char, y=y_lbl, xref="x", yref="y",
                 text=char_label, showarrow=False,
                 font=dict(color=YELL, size=11), xanchor="left"),
        ],
        legend=dict(orientation="h", y=1.12, x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=300,
    )
    return fig


def make_bode_phase(sys_label: str, f_test: float,
                    f_c: float, f_n: float,
                    f_char: float, char_label: str) -> go.Figure:
    f_lo  = max(f_char / 200.0, 0.001)
    f_hi  = min(f_char * 200.0, 1e5)
    f_arr = np.logspace(np.log10(f_lo), np.log10(f_hi), 800)

    _, phase_deg = freq_response(sys_label, f_arr, f_c, f_n)
    _, ph_test   = freq_response(sys_label, np.array([f_test]), f_c, f_n)
    ph_test = float(ph_test[0])

    ph_min = float(np.min(phase_deg)) - 10
    ph_max = float(np.max(phase_deg)) + 10
    y_lbl  = ph_max - 0.08 * (ph_max - ph_min)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_arr, y=phase_deg, mode="lines",
                             name="Phase [°]", line=dict(color=ACC2, width=2.5)))
    fig.add_trace(go.Scatter(x=[f_test], y=[ph_test], mode="markers",
                             name=f"f_test = {f_test:.2f} Hz",
                             marker=dict(color=ACC3, size=11, symbol="circle")))
    fig.update_layout(
        **_BASE,
        title=dict(text="Bode Plot – Phase", font=dict(size=15)),
        xaxis=dict(type="log", title="Frequency [Hz]", **_AX),
        yaxis=dict(title="Phase [°]", range=[ph_min, ph_max], **_AX),
        shapes=[
            dict(type="line", x0=f_test, x1=f_test, y0=ph_min, y1=ph_max,
                 xref="x", yref="y", line=dict(color=ACC3, width=1.5, dash="dot")),
            dict(type="line", x0=f_char, x1=f_char, y0=ph_min, y1=ph_max,
                 xref="x", yref="y", line=dict(color=YELL, width=1.5, dash="dash")),
        ],
        annotations=[
            dict(x=f_char, y=y_lbl, xref="x", yref="y",
                 text=char_label, showarrow=False,
                 font=dict(color=YELL, size=11), xanchor="left"),
        ],
        legend=dict(orientation="h", y=1.12, x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=300,
    )
    return fig


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
            st.session_state["f_n"] = fn

        st.divider()
        st.subheader("Test Frequency")
        f_test = st.slider("f_test [Hz]",
                           FT_MIN, FT_MAX, float(safe_get("f_test_hz")), FT_STEP)
        st.session_state["f_test_hz"] = f_test

        st.divider()
        st.subheader("💾 Save / Load")
        if st.button("💾 Save JSON", use_container_width=True):
            data = {k: safe_get(k) for k in DEFAULTS}
            st.download_button("⬇️ Download",
                               data=json.dumps(data, indent=2),
                               file_name="freq_response_params.json",
                               mime="application/json",
                               use_container_width=True)
        st.markdown("**📂 Load JSON**")
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
    f_c    = float(safe_get("f_c"))
    f_n    = float(safe_get("f_n"))
    f_test = float(safe_get("f_test_hz"))
    is_2nd = "2nd" in sys_label

    gain_lin, phase_deg = point_response(sys_label, f_test, f_c, f_n)
    gain_db = 20.0 * np.log10(max(gain_lin, 1e-12))

    if is_2nd:
        f_char     = f_n
        char_label = f"f_n = {f_n:.2f} Hz"
        char_str   = f"f_n = {f_n:.2f} Hz"
        ratio      = f_test / max(f_n, 1e-9)
    else:
        f_char     = f_c
        char_label = f"f_c = {f_c:.2f} Hz"
        char_str   = f"f_c = {f_c:.2f} Hz"
        ratio      = f_test / max(f_c, 1e-9)

    if ratio < 0.1:
        regime = "well below the characteristic frequency – signal passes almost unchanged"
    elif ratio < 0.7:
        regime = "in the transition band – moderate attenuation / boost"
    elif ratio < 1.4:
        regime = "near the characteristic frequency – strong filter effect"
    else:
        regime = "well above the characteristic frequency – strong attenuation / boost"

    st.info(
        f"**{sys_label}** | {char_str}\n\n"
        f"Test frequency **{f_test:.3f} Hz** is {regime}.\n"
        f"Output amplitude: **{gain_lin:.4f}×** the input "
        f"(**{gain_db:+.1f} dB**), phase shift: **{phase_deg:.1f}°**."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Gain [dB]",           f"{gain_db:+.2f} dB")
    c2.metric("Gain [linear ratio]",  f"{gain_lin:.4f}")
    c3.metric("Phase Shift",          f"{phase_deg:.1f}°")

    st.divider()
    st.plotly_chart(make_time_plot(sys_label, f_test, f_c, f_n),
                    use_container_width=True)
    st.plotly_chart(make_bode_magnitude(sys_label, f_test, f_c, f_n,
                                        f_char, char_label),
                    use_container_width=True)
    st.plotly_chart(make_bode_phase(sys_label, f_test, f_c, f_n,
                                    f_char, char_label),
                    use_container_width=True)


def tab_explanation() -> None:
    st.header("📖 Frequency Response & Bode Plots")

    st.markdown("""
### What is Frequency Response?

A linear system responds to a sinusoidal input with a sinusoidal output at the **same frequency**.
Only two things change: the **amplitude** (gain) and the **timing** (phase shift).

The **frequency response** captures how gain and phase vary across all frequencies.
A **Bode plot** shows this on a logarithmic frequency axis — giving the full picture at a glance.

---

### The Four System Types

| System | What it does | Key parameter |
|---|---|---|
| **1st Order Low-Pass** | Passes low frequencies, attenuates high ones | Cutoff frequency f_c |
| **1st Order High-Pass** | Passes high frequencies, attenuates low ones | Cutoff frequency f_c |
| **2nd Order Low-Pass** | Steeper roll-off, smooth transition | Natural frequency f_n |
| **2nd Order High-Pass** | High-freq emphasis, smooth transition | Natural frequency f_n |

---

### Transfer Functions H(jω)

Let ω = 2π · f, ω_c = 2π · f_c, ω_n = 2π · f_n.

**1st Order Low-Pass**
""")
    st.latex(r"H(j\omega) = \frac{1}{1 + j\,\omega/\omega_c}")
    st.markdown("At f = f_c: gain = 1/√2 ≈ **−3 dB**, phase = **−45°**")

    st.markdown("**1st Order High-Pass**")
    st.latex(r"H(j\omega) = \frac{j\,\omega/\omega_c}{1 + j\,\omega/\omega_c}")
    st.markdown("At f = f_c: gain = 1/√2 ≈ **−3 dB**, phase = **+45°**")

    st.markdown("**2nd Order Low-Pass**")
    st.latex(r"H(j\omega) = \frac{\omega_n^2}{\omega_n^2 - \omega^2 + 2j\,\xi\,\omega_n\,\omega}")
    st.markdown("Steeper roll-off than 1st order (−40 dB/decade). Phase shifts from 0° to −180°.")

    st.markdown("**2nd Order High-Pass**")
    st.latex(r"H(j\omega) = \frac{-\omega^2}{\omega_n^2 - \omega^2 + 2j\,\xi\,\omega_n\,\omega}")
    st.markdown("Steeper rise than 1st order (+40 dB/decade). Phase shifts from +180° to 0°.")

    st.markdown("""
---

### Reading a Bode Plot

- **Magnitude plot**: gain in dB vs. log frequency.
  0 dB = output equals input, −20 dB = output is 10× smaller.
- **Phase plot**: phase shift in degrees vs. log frequency.
- **🔴 Orange dot & dotted line**: current test frequency.
- **🟡 Yellow dashed line**: characteristic frequency (f_c or f_n).

---

### The −3 dB Cutoff

For a 1st-order filter, at f = f_c:
- Gain = 1/√2 ≈ 0.707 → **−3 dB**
- Phase shift = exactly **±45°**

For a 2nd-order system, f_n is the **natural frequency** – the transition region is centred there.
The roll-off is twice as steep: **−40 dB/decade** instead of −20 dB/decade.
""")


def tab_exercises() -> None:
    st.header("✏️ Exercises")
    st.markdown("Use the **sidebar** to set parameters and the **Single Frequency** tab to read results.")

    with st.expander("🔵 Exercise 1 – The −3 dB Point (1st Order)", expanded=True):
        st.markdown("""
1. Select **1st Order Low-Pass**, set f_c = 2 Hz.
2. Set f_test = 2 Hz (equal to f_c).
3. Read off the three metrics.

**Questions**
- What is the gain in dB? What is the linear ratio?
- What is the phase shift?
- Move f_test to 0.2 Hz, then to 20 Hz. How do the metrics change?

**Expected:** Gain ≈ −3 dB (ratio ≈ 0.707), phase ≈ −45° at f_test = f_c.
""")

    with st.expander("🔵 Exercise 2 – Low-Pass vs. High-Pass (1st Order)", expanded=False):
        st.markdown("""
1. Select **1st Order Low-Pass**, f_c = 2 Hz, f_test = 2 Hz.
2. Switch to **1st Order High-Pass** (keep same parameters).

**Questions**
- Is the gain the same for both types at f_test = f_c?
- How does the phase sign differ?
- At very low frequencies, which filter passes the signal? At very high frequencies?
""")

    with st.expander("🟢 Exercise 3 – Steeper Roll-Off (2nd Order Low-Pass)", expanded=False):
        st.markdown("""
1. Select **1st Order Low-Pass**, f_c = 5 Hz.
2. Note the Bode magnitude slope above f_c.
3. Now switch to **2nd Order Low-Pass**, f_n = 5 Hz.

**Questions**
- How does the slope above f_n compare to the 1st order case?
- Is the transition region wider or narrower?
- Set f_test to 10 Hz, then 50 Hz. Compare the gain values between both filter types.

**Expected:** 2nd order rolls off at −40 dB/decade vs. −20 dB/decade for 1st order.
""")

    with st.expander("🟢 Exercise 4 – Phase Shift of 2nd Order Systems", expanded=False):
        st.markdown("""
1. Select **2nd Order Low-Pass**, f_n = 5 Hz.
2. Set f_test well below f_n (e.g. 0.5 Hz), then at f_n, then well above (e.g. 50 Hz).

**Questions**
- What is the phase at very low frequencies?
- What is the phase exactly at f_n?
- What is the total phase shift range for a 2nd order low-pass?

**Expected:** Phase goes from 0° → −90° at f_n → −180° — twice the range of a 1st order filter.
""")

    with st.expander("🟢 Exercise 5 – 2nd Order High-Pass", expanded=False):
        st.markdown("""
1. Select **2nd Order High-Pass**, f_n = 5 Hz.
2. Sweep f_test from 0.5 Hz to 50 Hz.

**Questions**
- Below f_n, how steeply does the gain fall off (dB per decade)?
- At very high frequencies, what does the gain approach?
- Compare the phase curve with the 2nd Order Low-Pass at the same f_n.
""")

    with st.expander("📐 Exercise 6 – Predicting the Output Signal", expanded=False):
        st.markdown("""
1. Select **1st Order Low-Pass**, f_c = 1 Hz, f_test = 3 Hz.
2. Note the **Gain [linear ratio]** and **Phase Shift** values.

**Task:** The input signal is:

> u(t) = sin(2π · 3 · t)

Write the expected output y(t) as a formula using the values you read off.
Verify it matches the time-domain plot.
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
