import streamlit as st
import numpy as np
from scipy import signal as scipy_signal
import plotly.graph_objects as go
import pandas as pd
import json

# ==========================================
# --- 0. PAGE CONFIG & STATE MANAGEMENT ---
# ==========================================
st.set_page_config(page_title="Signals & Waves", layout="wide", page_icon="🌊")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size: 18px !important; font-weight: bold !important; }
    .stMetric { background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #303030; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
SIM_DURATION     = 3.0    # Visible time window [s]
SIM_DT_CONT      = 0.001  # Time step for continuous mode [s]
DISCRETE_MIN_PTS = 10
DISCRETE_MAX_PTS = 300

# Valid option lists — the single source of truth for all dropdowns/radios.
# Any value read from session state or JSON is validated against these.
SIGNAL_OPTIONS = ["Sine", "Square", "Sawtooth"]
TYPE_OPTIONS   = ["Continuous", "Discrete"]

# Defaults
DEFAULTS: dict = {
    "sig_form":    "Sine",
    "sig_type":    "Continuous",
    "sig_freq":    1.0,
    "sig_amp":     5.0,
    "sig_offset":  0.0,
    "sig_phase":   0.0,
    "sig_samples": 50,
    "sig_noise":   0.0,
}


def init_state() -> None:
    """Initialize every session-state key with its default value if not present."""
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def safe_get(key: str):
    """Return session-state value, falling back to the default if missing."""
    return st.session_state.get(key, DEFAULTS[key])


def clamp(value, lo, hi):
    """Clamp a numeric value to [lo, hi]."""
    return max(lo, min(hi, value))


# ==========================================
# --- 1. CALLBACKS ---
# ==========================================
def on_upload_callback() -> None:
    """Load a JSON config file and validate every value before applying it."""
    uploader = st.session_state.get("json_uploader")
    if uploader is None:
        return
    try:
        data = json.load(uploader)
    except (json.JSONDecodeError, Exception) as exc:
        st.error(f"Could not read the file – is it a valid JSON file? ({exc})")
        return

    # Apply values one-by-one with type coercion and range validation
    mapping = {
        "sig_form":    (str,   lambda v: v if v in SIGNAL_OPTIONS else DEFAULTS["sig_form"]),
        "sig_type":    (str,   lambda v: v if v in TYPE_OPTIONS   else DEFAULTS["sig_type"]),
        "sig_freq":    (float, lambda v: clamp(float(v),  0.1,   10.0)),
        "sig_amp":     (float, lambda v: clamp(float(v),  0.1,   10.0)),
        "sig_offset":  (float, lambda v: clamp(float(v), -10.0,  10.0)),
        "sig_phase":   (float, lambda v: clamp(float(v), -180.0, 180.0)),
        "sig_samples": (int,   lambda v: clamp(int(v), DISCRETE_MIN_PTS, DISCRETE_MAX_PTS)),
        "sig_noise":   (float, lambda v: clamp(float(v),  0.0,   5.0)),
    }
    for key, (_, validator) in mapping.items():
        if key in data:
            try:
                st.session_state[key] = validator(data[key])
            except Exception:
                st.session_state[key] = DEFAULTS[key]  # fall back on any error

    st.toast("✅ Settings loaded!", icon="💾")


# ==========================================
# --- 2. SIGNAL GENERATION ---
# ==========================================
# NOTE: @st.cache_data receives only primitive types (str, float, int, bool).
# Passing a dataclass or any custom object can cause cache-hashing issues across
# Streamlit versions — primitives are always safe.

@st.cache_data(show_spinner=False)
def generate_signal(
    form:      str,
    sig_type:  str,
    frequency: float,
    amplitude: float,
    offset:    float,
    phase_deg: float,
    samples:   int,
    noise_amp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) arrays for the requested signal configuration.

    Continuous mode: uniform 1 ms time steps over SIM_DURATION seconds.
    Discrete mode:   'samples' evenly-spaced points over the same window.

    All parameters are clamped here as a second safety layer in case the
    caller passes an out-of-range value.
    """
    frequency = max(frequency, 0.01)          # guard against zero / negative
    amplitude = max(amplitude, 0.0)
    samples   = max(samples, 2)               # need at least 2 points

    if sig_type == "Continuous":
        t = np.arange(0.0, SIM_DURATION, SIM_DT_CONT)
    else:
        t = np.linspace(0.0, SIM_DURATION, samples)

    phase_rad = phase_deg * np.pi / 180.0
    arg       = 2.0 * np.pi * frequency * t + phase_rad

    if form == "Sine":
        y = amplitude * np.sin(arg) + offset
    elif form == "Square":
        y = amplitude * scipy_signal.square(arg) + offset
    elif form == "Sawtooth":
        y = amplitude * scipy_signal.sawtooth(arg) + offset
    else:
        # Unknown form — return flat line so the app never crashes
        y = np.full_like(t, offset)

    # Gaussian noise — fixed seed so the result is deterministic and cache-safe.
    # The seed is combined with a hash of the relevant parameters so that
    # changing any setting still gives a fresh noise pattern visually.
    if noise_amp > 0.0:
        seed = hash((form, round(frequency, 3), round(noise_amp, 3))) % (2**31)
        rng  = np.random.default_rng(seed=seed)
        y    = y + rng.normal(0.0, noise_amp, len(t))

    return t, y


# ==========================================
# --- 3. PLOT HELPERS ---
# ==========================================
def build_stem_traces(t: np.ndarray, y: np.ndarray):
    """Return (stem_lines_trace, markers_trace) for a discrete stem plot.

    Uses a single Scatter trace with None separators instead of one add_shape
    call per sample — far more efficient for large sample counts.
    """
    # Build interleaved stem lines: [x0, x0, None, x1, x1, None, ...]
    x_lines: list = []
    y_lines: list = []
    for xv, yv in zip(t.tolist(), y.tolist()):
        x_lines += [xv, xv, None]
        y_lines += [0.0, yv, None]

    stem_trace = go.Scatter(
        x=x_lines, y=y_lines,
        mode="lines",
        line=dict(color="#EF553B", width=1.5),
        hoverinfo="skip",
        showlegend=False,
    )
    dot_trace = go.Scatter(
        x=t, y=y,
        mode="markers",
        name="Sample points",
        marker=dict(color="#EF553B", size=8),
    )
    return stem_trace, dot_trace


def apply_dark_axes(fig: go.Figure) -> None:
    """Apply consistent dark-theme axis styling."""
    shared = dict(
        showgrid=True, gridwidth=1, gridcolor="#333333",
        zeroline=True, zerolinewidth=2, zerolinecolor="#888888",
    )
    fig.update_xaxes(**shared)
    fig.update_yaxes(**shared)


# ==========================================
# --- 4. SIDEBAR ---
# ==========================================
def render_sidebar() -> None:
    """Render the full sidebar. Called before st.tabs() so it is always visible."""
    with st.sidebar:

        # ── Signal shape & type ──────────────────────────────────────────
        st.header("1. Choose a Signal")

        # Plain selectbox — no format_func, no dict lookup.
        # The stored session-state value is always one of SIGNAL_OPTIONS.
        sig_form_idx = SIGNAL_OPTIONS.index(safe_get("sig_form"))
        st.selectbox(
            "Signal shape:",
            options=SIGNAL_OPTIONS,
            index=sig_form_idx,
            key="sig_form",
            help=(
                "Sine: smooth wave  |  "
                "Square: hard on/off  |  "
                "Sawtooth: slow rise, sudden drop"
            ),
        )

        type_idx = TYPE_OPTIONS.index(safe_get("sig_type"))
        st.radio(
            "How is the signal recorded?",
            options=TYPE_OPTIONS,
            index=type_idx,
            key="sig_type",
            horizontal=True,
            help=(
                "Continuous = smooth line at every moment in time.  "
                "Discrete = only measured at specific sample points (like a digital sensor)."
            ),
        )

        st.divider()

        # ── Signal properties ────────────────────────────────────────────
        st.header("2. Shape the Signal")
        st.slider(
            "How fast does it oscillate? (Frequency in Hz)",
            min_value=0.1, max_value=10.0, step=0.1,
            key="sig_freq",
            help="1 Hz = 1 full oscillation per second.",
        )
        st.slider(
            "How high does it swing? (Amplitude)",
            min_value=0.1, max_value=10.0, step=0.1,
            key="sig_amp",
            help="The maximum deflection of the signal above and below the baseline.",
        )
        st.slider(
            "Shift the baseline up or down (DC Offset)",
            min_value=-10.0, max_value=10.0, step=0.5,
            key="sig_offset",
            help="Moves the entire signal up or down without changing its shape.",
        )
        st.slider(
            "Shift the signal in time (Phase in °)",
            min_value=-180.0, max_value=180.0, step=5.0,
            key="sig_phase",
            help="0°: starts at zero and rises.  90°: starts at the peak.  -90°: starts at the valley.",
        )

        st.divider()

        # ── Measurement settings ─────────────────────────────────────────
        st.header("3. Measurement Settings")
        if safe_get("sig_type") == "Discrete":
            st.slider(
                "How many measurement points? (Samples)",
                min_value=DISCRETE_MIN_PTS,
                max_value=DISCRETE_MAX_PTS,
                step=5,
                key="sig_samples",
                help=(
                    "Fewer samples = coarser picture.  "
                    "Too few samples can make the signal look completely different (aliasing)."
                ),
            )
        st.slider(
            "Add sensor noise (σ)",
            min_value=0.0, max_value=5.0, step=0.1,
            key="sig_noise",
            help="Simulates the random jitter a real sensor produces.",
        )

        st.divider()

        # ── Save / load ──────────────────────────────────────────────────
        st.header("💾 Settings")
        keys_to_save = list(DEFAULTS.keys())
        conf_data    = {k: safe_get(k) for k in keys_to_save}

        st.download_button(
            label="📥 Save settings (JSON)",
            data=json.dumps(conf_data, indent=2),
            file_name=f"signal_config_{safe_get('sig_form').lower()}.json",
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
    # Always initialise state first — before any widget renders.
    init_state()

    st.title("🌊 Signals & Waves – Explore How Signals Work")
    st.markdown(
        "Every real-world measurement – temperature, speed, voltage – is a signal.  "
        "Here you can build your own and explore how each property changes its shape."
    )

    # Sidebar is rendered before st.tabs() so it appears on every tab.
    render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "1. 🎛️ Try It Out",
        "2. 📘 Explanation",
        "3. 📝 Exercises",
    ])

    # ── TAB 1 ─────────────────────────────────────────────────────────────
    with tab1:

        # Read state through safe_get so missing keys never raise KeyError.
        t, y = generate_signal(
            form      = safe_get("sig_form"),
            sig_type  = safe_get("sig_type"),
            frequency = float(safe_get("sig_freq")),
            amplitude = float(safe_get("sig_amp")),
            offset    = float(safe_get("sig_offset")),
            phase_deg = float(safe_get("sig_phase")),
            samples   = int(safe_get("sig_samples")),
            noise_amp = float(safe_get("sig_noise")),
        )

        # --- 3 key metrics ---
        y_max = float(np.max(y))
        y_min = float(np.min(y))
        y_pp  = y_max - y_min

        c1, c2, c3 = st.columns(3)
        c1.metric("Highest Point",          f"{y_max:.2f}",
                  help="The maximum value the signal reaches.")
        c2.metric("Lowest Point",           f"{y_min:.2f}",
                  help="The minimum value the signal reaches.")
        c3.metric("Total Swing (Peak-Peak)", f"{y_pp:.2f}",
                  help="Distance from lowest to highest point. For a pure sine: 2 × Amplitude.")

        st.markdown("---")

        # --- Plot ---
        fig = go.Figure()

        if safe_get("sig_type") == "Continuous":
            fig.add_trace(go.Scatter(
                x=t, y=y,
                mode="lines",
                name="Signal y(t)",
                line=dict(color="#00CC96", width=3),
            ))
        else:
            stem_trace, dot_trace = build_stem_traces(t, y)
            fig.add_trace(stem_trace)
            fig.add_trace(dot_trace)

        margin = max((y_max - y_min) * 0.2, 1.0)
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude y(t)",
            yaxis_range=[y_min - margin, y_max + margin],
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_dark_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        # --- CSV download ---
        df  = pd.DataFrame({"Time [s]": t, "Amplitude y(t)": y})
        csv = df.to_csv(index=False).encode("utf-8")
        _, col_btn = st.columns([3, 1])
        with col_btn:
            st.download_button(
                label="📊 Download data (CSV)",
                data=csv,
                file_name=f"signal_{safe_get('sig_form').lower()}_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── TAB 2 ─────────────────────────────────────────────────────────────
    with tab2:
        st.header("📘 What Is a Signal – and What Shapes It?")
        st.markdown("""
> **Key idea:** A signal is any quantity that changes over time –
> temperature, microphone voltage, motor speed. Every signal is described by four basic properties.
        """)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📐 Amplitude – How High Does It Swing?")
            st.markdown("""
The amplitude is the maximum deflection above and below the centre line.

**Everyday analogy:** Think of a swing. A gentle push → barely moves.
A strong push → swings far. Amplitude = how far it goes.

- Double the amplitude → signal reaches twice as high *and* twice as low
- Does not affect speed or shape, only size
            """)

            st.subheader("⏱️ Frequency – How Fast Does It Oscillate?")
            st.markdown("""
Frequency = how many full oscillations happen per second. Unit: **Hertz (Hz)**.

**Everyday analogy:** Pluck a guitar string.
Low note → vibrates slowly (few Hz).  High note → vibrates fast (many Hz).

- 1 Hz = 1 full cycle per second
- 10 Hz = 10 cycles per second (10× faster)
            """)

        with col_b:
            st.subheader("📏 DC Offset – Where Is the Centre Line?")
            st.markdown("""
The DC offset shifts the whole signal up or down without changing its shape.

**Everyday analogy:** Room temperature oscillates between 19 °C and 21 °C.
The centre (20 °C) is the offset – the baseline around which it swings.

- Offset 0 → signal swings symmetrically around zero
- Offset +5 → same swing, but centred on 5
            """)

            st.subheader("⏩ Phase – Where Does It Start?")
            st.markdown("""
Phase shifts the signal in time, setting where in its cycle it begins at t = 0.

**Everyday analogy:** Two runners in the same race, but one has a head start.
Same speed, never at the same position – they are "out of phase".

- 0°: sine starts at zero and rises
- 90°: sine starts at its peak
- −90°: sine starts at its lowest point
            """)

        st.divider()

        st.subheader("🔵 Continuous vs. 🔴 Discrete Signals")
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("""
**Continuous:** Exists at every single moment – a smooth, unbroken line.

Like an analogue thermometer: the needle moves continuously and always
shows the exact current temperature.
            """)
        with col_d:
            st.markdown("""
**Discrete:** Only exists at specific moments – the sample points.
No value exists in between.

Like a digital thermometer that updates once per second – all the values
between updates are missing.

**Important:** Too few samples and the signal looks completely different
from the original. This effect is called **aliasing**.
            """)

        st.divider()

        st.subheader("📊 The Three Signal Shapes")
        col_e, col_f, col_g = st.columns(3)
        with col_e:
            st.markdown("""
**🟢 Sine**

The most natural oscillation.
Sound waves, AC voltage, pendulums.
Smooth, round, no sharp edges.
            """)
        with col_f:
            st.markdown("""
**🟠 Square**

Switches between exactly +A and −A – nothing in between.
Common in digital circuits: on/off, high/low.
            """)
        with col_g:
            st.markdown("""
**🔵 Sawtooth**

Rises steadily, drops instantly.
Used in oscilloscopes and audio synthesisers.
            """)

    # ── TAB 3 ─────────────────────────────────────────────────────────────
    with tab3:
        st.header("📝 Exercises")
        st.markdown(
            "Use the **left sidebar** to adjust the signal and observe how the plot changes."
        )

        with st.expander("📐 Exercise 1 – Amplitude", expanded=True):
            st.markdown("""
**Task:** See how amplitude controls the size of the signal.

**Start:** Sine · Frequency **1 Hz** · Offset **0** · Phase **0°** · no noise

1. Set Amplitude to **2** → note Highest Point and Total Swing
2. Set Amplitude to **8** → both values should have doubled

The shape stays the same – only the size changes.

→ *Amplitude controls size, nothing else.*
            """)

        with st.expander("⏱️ Exercise 2 – Frequency"):
            st.markdown("""
**Task:** See how frequency controls the speed of oscillation.

**Start:** Sine · Amplitude **5** · Offset **0** · Phase **0°**

1. Frequency **1 Hz** → count 3 full waves in the 3-second window
2. Frequency **3 Hz** → 9 waves (3× as many)
3. Frequency **0.5 Hz** → only 1.5 waves fit

→ *Frequency = oscillations per second. Higher → more waves, same time.*
            """)

        with st.expander("📏 Exercise 3 – DC Offset"):
            st.markdown("""
**Task:** See how the offset shifts the signal without changing its shape.

**Start:** Sine · Frequency **1 Hz** · Amplitude **3**

1. Offset **0** → swings between −3 and +3 (Total Swing = 6)
2. Offset **+5** → swings between +2 and +8 (Total Swing still 6)
3. Offset **−5** → swings between −8 and −2

→ *Offset moves the signal up or down. Amplitude and frequency stay unchanged.*
            """)

        with st.expander("⏩ Exercise 4 – Phase Shift"):
            st.markdown("""
**Task:** Discover how phase moves the signal in time.

**Start:** Sine · Frequency **1 Hz** · Amplitude **5** · Offset **0**

1. Phase **0°** → starts at zero, rises immediately
2. Phase **90°** → starts at the peak (+5). This is a cosine!
3. Phase **−90°** → starts at the valley (−5)
4. Phase **180°** → signal is completely flipped (equivalent to ×−1)

→ *Phase shifts timing only – size and speed are unchanged.*
            """)

        with st.expander("🔴 Exercise 5 – Continuous vs. Discrete (Aliasing)"):
            st.markdown("""
**Task:** See what happens when a signal is sampled too slowly.

**Start:** Sine · Frequency **2 Hz** · Amplitude **5** → switch to **Discrete**

1. Samples **100** → dots follow the sine closely ✅
2. Samples **20** → fewer dots, still recognisable
3. Samples **10** → only 10 points for 6 full oscillations → shape starts to distort
4. Samples **5** → looks like a completely different signal! This is **aliasing**.

→ *Sample at least twice as fast as the signal frequency (Nyquist rule).*
            """)

        with st.expander("🎲 Exercise 6 – Sensor Noise"):
            st.markdown("""
**Task:** Understand how noise affects measurement quality.

**Start:** Sine · Frequency **1 Hz** · Amplitude **5** · Offset **0** · Continuous

1. Noise σ = **0** → perfectly clean sine
2. Noise σ = **0.5** → slight shimmer, shape still clear
3. Noise σ = **2.0** → noticeably noisy but still visible
4. Noise σ = **5.0** → hard to recognise the original sine

When noise equals amplitude (both = 5), the signal-to-noise ratio is 1:1 – very poor.
A real sensor should have noise far below the signal amplitude.

→ *This ratio is called the **Signal-to-Noise Ratio (SNR)**.*
            """)


if __name__ == "__main__":
    main()
