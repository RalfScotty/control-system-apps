import streamlit as st
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from dataclasses import dataclass
import pandas as pd
import json

# ==========================================
# --- 0. PAGE CONFIG & STATE MANAGEMENT ---
# ==========================================
st.set_page_config(page_title="Signals & Systems", layout="wide", page_icon="🔲")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size: 18px !important; font-weight: bold !important; }
    .stMetric { background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #303030; }
</style>
""", unsafe_allow_html=True)

# --- Simulation constants ---
SIM_DT       = 0.001  # Time step [s]
SIM_DURATION = 3.0    # Simulation duration [s]

# Signal display labels (internal key → readable label)
SIGNAL_LABELS = {
    "Sine":     "Sine  –  smooth, regular oscillation",
    "Square":   "Square  –  jumps sharply between high and low",
    "Sawtooth": "Sawtooth  –  rises slowly, drops suddenly",
    "Step":     "Step  –  one-time jump from 0 to amplitude",
}

# System display labels (internal key → readable label)
SYSTEM_LABELS = {
    "Lowpass":   "🟢 Low-pass Filter  –  lets slow signals through",
    "Highpass":  "🔵 High-pass Filter  –  only lets changes through",
    "Nonlinear": "🟣 Non-linear System  –  reshapes the signal (y = x²)",
}


def init_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value


init_state("sig_form",   "Sine")
init_state("sig_freq",   1.0)
init_state("sig_amp",    5.0)
init_state("sig_offset", 0.0)
init_state("sig_noise",  0.0)
init_state("sys_sel",    "Lowpass")
init_state("sys_param",  5.0)


# ==========================================
# --- 1. CALLBACKS ---
# ==========================================
def on_upload_callback():
    if st.session_state.json_uploader is not None:
        try:
            data = json.load(st.session_state.json_uploader)
            for k, v in data.items():
                if k in st.session_state:
                    if isinstance(st.session_state[k], float):
                        st.session_state[k] = float(v)
                    elif isinstance(st.session_state[k], int):
                        st.session_state[k] = int(v)
                    else:
                        st.session_state[k] = str(v)
            st.toast("✅ Settings loaded!", icon="💾")
        except json.JSONDecodeError:
            st.error("Could not read the file – is it a valid JSON file?")
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# --- 2. DATA STRUCTURE ---
# ==========================================
@dataclass(frozen=True)  # frozen=True → hashable → correct cache keys for @st.cache_data
class SimConfig:
    form:          str
    frequency:     float
    amplitude:     float
    offset:        float
    noise_amp:     float
    sys_selection: str
    sys_parameter: float


# ==========================================
# --- 3. SIMULATION ---
# ==========================================
@st.cache_data(show_spinner=False)
def simulate_system(config: SimConfig):
    """Generates an input signal and passes it through the selected system."""
    fs  = 1.0 / SIM_DT
    nyq = 0.5 * fs
    t   = np.arange(0, SIM_DURATION, SIM_DT)

    # --- Generate input signal ---
    if config.form == "Sine":
        y_in = config.amplitude * np.sin(2 * np.pi * config.frequency * t) + config.offset
    elif config.form == "Square":
        y_in = config.amplitude * signal.square(2 * np.pi * config.frequency * t) + config.offset
    elif config.form == "Sawtooth":
        y_in = config.amplitude * signal.sawtooth(2 * np.pi * config.frequency * t) + config.offset
    elif config.form == "Step":
        y_in = np.full_like(t, config.offset)
        y_in[len(t) // 4:] += config.amplitude  # Step occurs at t = 0.75 s
    else:
        y_in = np.full_like(t, config.offset)

    # --- Add Gaussian noise (fixed seed → reproducible & cache-safe) ---
    if config.noise_amp > 0:
        rng  = np.random.default_rng(seed=42)
        y_in = y_in + rng.normal(0, config.noise_amp, len(t))

    # --- Apply the selected system ---
    y_out    = np.zeros_like(y_in)
    safe_cut = min(config.sys_parameter, nyq - 0.1)
    norm_cut = safe_cut / nyq

    if config.sys_selection == "Lowpass":
        # 2nd-order Butterworth low-pass filter
        sos = signal.butter(2, norm_cut, btype="low", analog=False, output="sos")
        zi  = signal.sosfilt_zi(sos) * y_in[0]
        y_out, _ = signal.sosfilt(sos, y_in, zi=zi)

    elif config.sys_selection == "Highpass":
        # 2nd-order Butterworth high-pass filter
        sos = signal.butter(2, norm_cut, btype="high", analog=False, output="sos")
        zi  = signal.sosfilt_zi(sos) * y_in[0]
        y_out, _ = signal.sosfilt(sos, y_in, zi=zi)

    elif config.sys_selection == "Nonlinear":
        # Squarer: y(t) = x(t)²
        y_out = y_in ** 2

    return t, y_in, y_out


# ==========================================
# --- 4. SIDEBAR ---
# ==========================================
def render_sidebar():
    with st.sidebar:
        # --- Signal selection ---
        st.header("1. Choose a Signal")
        st.selectbox(
            "What signal goes in?",
            options=list(SIGNAL_LABELS.keys()),
            format_func=lambda k: SIGNAL_LABELS[k],
            key="sig_form",
        )
        if st.session_state.sig_form != "Step":
            st.slider(
                "How fast does it oscillate? (Frequency in Hz)",
                0.5, 20.0, step=0.5, key="sig_freq",
                help="1 Hz = 1 full oscillation per second.",
            )
        st.slider(
            "How high should it swing? (Amplitude)",
            1.0, 10.0, step=1.0, key="sig_amp",
            help="The maximum deflection of the signal above and below zero.",
        )
        st.slider(
            "Shift the baseline (DC Offset)",
            -10.0, 10.0, step=0.5, key="sig_offset",
            help="Moves the entire signal up or down.",
        )
        st.slider(
            "Add sensor noise (σ)",
            0.0, 5.0, step=0.1, key="sig_noise",
            help="Simulates the random jitter a real sensor produces.",
        )

        st.divider()

        # --- System selection ---
        st.header("2. Choose a System")
        st.selectbox(
            "Which black box should the signal pass through?",
            options=list(SYSTEM_LABELS.keys()),
            format_func=lambda k: SYSTEM_LABELS[k],
            key="sys_sel",
        )

        sel = st.session_state.sys_sel
        if sel in ("Lowpass", "Highpass"):
            lbl = (
                "Let through up to … (Cutoff frequency in Hz)"
                if sel == "Lowpass"
                else "Let through from … (Cutoff frequency in Hz)"
            )
            st.slider(lbl, 0.1, 20.0, step=0.1, key="sys_param",
                      help="The frequency at which the system switches between passing and blocking.")
            # Warn when cutoff ≈ signal frequency (–3 dB edge case)
            fc, fs = st.session_state.sys_param, st.session_state.sig_freq
            if st.session_state.sig_form != "Step" and abs(fc - fs) < 0.3:
                st.info("ℹ️ Cutoff ≈ signal frequency → the signal is reduced by half (–3 dB).")
        else:
            st.caption("This system has no adjustable parameter.")

        st.divider()

        # --- Save / load configuration ---
        st.header("💾 Settings")
        keys_to_save = ["sig_form", "sig_freq", "sig_amp", "sig_offset",
                        "sig_noise", "sys_sel", "sys_param"]
        current_conf = {k: st.session_state[k] for k in keys_to_save}

        st.download_button(
            label="📥 Save settings (JSON)",
            data=json.dumps(current_conf, indent=2),
            file_name="system_config.json",
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
def main():
    st.title("🔲 Signals & Systems – What Does a Black Box Do?")
    st.markdown(
        "You already know what signals look like. "
        "Now let's explore **what happens when a signal passes through a system** – "
        "without needing to know what is inside."
    )

    render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "1. 🚀 Try It Out",
        "2. 📘 Explanation",
        "3. 📝 Exercises",
    ])

    # ── TAB 1: Try it out ─────────────────────────────────────────────────
    with tab1:

        # --- Black box diagram ---
        st.markdown("### The Black Box Principle")
        c_in, c_arr1, c_box, c_arr2, c_out = st.columns([2.2, 0.4, 1.6, 0.4, 2.2])
        with c_in:
            st.markdown(
                """<div style='border:2px solid #00CC96; border-radius:10px; padding:12px; text-align:center;'>
                <b style='color:#00CC96'>📥 Input Signal x(t)</b><br>
                The signal we feed in
                </div>""", unsafe_allow_html=True
            )
        with c_arr1:
            st.markdown(
                "<div style='text-align:center; font-size:2.5rem; padding-top:10px'>→</div>",
                unsafe_allow_html=True
            )
        with c_box:
            sys_name = SYSTEM_LABELS[st.session_state.sys_sel].split("–")[0].strip()
            st.markdown(
                f"""<div style='border:2px solid #888888; border-radius:10px; padding:12px; text-align:center; background:#1a1a2e;'>
                <b style='color:#AAAAAA'>🔲 BLACK BOX</b><br>
                <small>{sys_name}</small>
                </div>""", unsafe_allow_html=True
            )
        with c_arr2:
            st.markdown(
                "<div style='text-align:center; font-size:2.5rem; padding-top:10px'>→</div>",
                unsafe_allow_html=True
            )
        with c_out:
            st.markdown(
                """<div style='border:2px solid #FF9F1C; border-radius:10px; padding:12px; text-align:center;'>
                <b style='color:#FF9F1C'>📤 Output Signal y(t)</b><br>
                What the system makes of it
                </div>""", unsafe_allow_html=True
            )

        st.markdown("---")

        # --- Run simulation ---
        config = SimConfig(
            form=st.session_state.sig_form,
            frequency=st.session_state.sig_freq,
            amplitude=st.session_state.sig_amp,
            offset=st.session_state.sig_offset,
            noise_amp=st.session_state.sig_noise,
            sys_selection=st.session_state.sys_sel,
            sys_parameter=st.session_state.sys_param,
        )
        t, y_in, y_out = simulate_system(config)

        # --- Metrics (3 cards) ---
        amp_in  = float(np.max(np.abs(y_in)))
        amp_out = float(np.max(np.abs(y_out)))

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Input Amplitude",
            f"{amp_in:.2f}",
            help="Peak absolute value of the input signal.",
        )
        c2.metric(
            "Output Amplitude",
            f"{amp_out:.2f}",
            help="Peak absolute value of the output signal.",
        )

        # Attenuation only makes sense for linear systems
        if config.sys_selection in ("Lowpass", "Highpass") and amp_in > 1e-9:
            atten = (1.0 - amp_out / amp_in) * 100.0
            label = "Attenuation" if atten >= 0 else "Amplification"
            c3.metric(
                label,
                f"{abs(atten):.1f} %",
                help="How much the system weakened (positive) or amplified (negative) the signal.",
            )
        else:
            c3.metric(
                "System Effect",
                "–",
                help="Attenuation % is not meaningful for non-linear systems "
                     "because the output amplitude depends non-linearly on the input.",
            )

        st.markdown("---")

        # --- Time-domain plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=y_in,
            mode="lines", name="x(t)  Input",
            line=dict(color="#00CC96", width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=t, y=y_out,
            mode="lines", name="y(t)  Output",
            line=dict(color="#FF9F1C", width=4),
        ))

        y_all  = np.concatenate([y_in, y_out])
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        margin = (y_max - y_min) * 0.15 if y_max != y_min else 1.0

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            yaxis_range=[y_min - margin, y_max + margin],
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#333333",
                         zeroline=True, zerolinewidth=2, zerolinecolor="#888888")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#333333",
                         zeroline=True, zerolinewidth=2, zerolinecolor="#888888")

        st.plotly_chart(fig, use_container_width=True)

        # --- CSV download ---
        df  = pd.DataFrame({"Time [s]": t, "Input x(t)": y_in, "Output y(t)": y_out})
        csv = df.to_csv(index=False).encode("utf-8")
        _, col_btn = st.columns([3, 1])
        with col_btn:
            st.download_button(
                label="📊 Download data (CSV)",
                data=csv,
                file_name="simulation_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── TAB 2: Explanation ─────────────────────────────────────────────────
    with tab2:
        st.header("📘 What Is Inside the Black Box?")

        st.markdown("""
> **Key idea:** A system takes a signal in and gives back a changed signal.
> Whether and how it changes depends on which system you choose.
        """)

        st.divider()

        # --- System 1 ---
        st.subheader("🟢 System 1 – Low-pass Filter")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("""
**Everyday analogy:** Imagine calling someone and talking very fast.
The listener only catches the calm, low tone of your voice – the fast, hectic parts get lost.

**What does a low-pass filter do?**
- Slow oscillations (low frequencies) → **let through** ✅
- Fast oscillations (high frequencies) → **blocked** ❌
- Noise is always very fast → **filtered out**

The **cutoff frequency** sets the boundary:
everything below passes, everything above is weakened.
            """)
        with col_b:
            st.info("""
**Try it (Tab 1):**
1. Set Sine at 2 Hz, noise σ = 2.0
2. Set cutoff frequency to **3 Hz**
3. The noise almost disappears – the clean sine remains!

The low-pass filter acts like a **smoothing tool** –
it removes rapid jitter and keeps the overall shape.
            """)

        st.divider()

        # --- System 2 ---
        st.subheader("🔵 System 2 – High-pass Filter")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("""
**Everyday analogy:** A motion sensor does not react when you stand still.
But as soon as you move quickly – alarm! It only registers **changes**.

**What does a high-pass filter do?**
- Slow signals and constant baselines (DC offset) → **blocked** ❌
- Fast changes and sudden jumps → **let through** ✅

The **cutoff frequency** defines how fast a change has to be to pass.
            """)
        with col_b:
            st.info("""
**Try it (Tab 1):**
1. Set Sine at 1 Hz, DC Offset to **+6**
2. Set cutoff frequency to **0.5 Hz**
3. The offset disappears – the signal swings around zero again!

The high-pass filter acts like an **AC coupler** –
it lets oscillations through but ignores the fixed baseline.
            """)

        st.divider()

        # --- System 3 ---
        st.subheader("🟣 System 3 – Non-linear System (y = x²)")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("""
**Everyday analogy:** Think of a heat lamp. If you double the power setting,
it does not become twice as warm – it gets much hotter.
The output does not grow in proportion to the input.

**What does this system do?**
- Every input value is **squared**: a negative value → positive output
- −3 gives the same result as +3: both become 9
- The output is **always positive**, regardless of the sign of the input
- The system can **create new frequencies** that were not there before!

**This is the key difference** from the two filters:
Low-pass and high-pass only change strength and timing.
A non-linear system can fundamentally reshape the signal.
            """)
        with col_b:
            st.info("""
**Try it (Tab 1):**

*Experiment 1 – Frequency doubling:*
Sine at **1 Hz**, Amplitude **3**, Offset **0**
→ The output oscillates at **2 Hz**!

*Experiment 2 – Everything becomes positive:*
Square at **1 Hz**, Amplitude **3**, Offset **0**
→ Input jumps between +3 and −3.
Squared: (+3)² = 9 and (−3)² = 9 → **flat line at 9**

*Experiment 3 – Parabola shape:*
Choose Sawtooth → a rising straight line becomes
a curved upward arc (a parabola)
            """)

        st.divider()

        st.subheader("🔑 The Key Difference: Linear vs. Non-linear")
        st.markdown("""
| | 🟢 Low-pass | 🔵 High-pass | 🟣 Squarer |
|---|---|---|---|
| **Can create new frequencies?** | No | No | **Yes!** |
| **Always proportional?** | Yes | Yes | **No** |
| **Negative input stays negative?** | Yes | Yes | **No** |
| **Type** | Linear | Linear | **Non-linear** |

**Rule of thumb:** With linear systems you can predict the output.
With non-linear systems, surprising things can happen –
which makes them both more complex and more interesting.
        """)

    # ── TAB 3: Exercises ───────────────────────────────────────────────────
    with tab3:
        st.header("📝 Exercises")
        st.markdown(
            "Adjust the parameters in the **left sidebar (Tab 1)** "
            "and observe how the orange output signal changes."
        )

        with st.expander("🟢 Low-pass – Remove noise from a signal", expanded=True):
            st.markdown("""
**Task:** A sensor measures a signal, but with a lot of noise. Can you clean it up?

**Settings:**
- Signal: **Sine**, Frequency **2 Hz**, Amplitude **5**
- Noise: **σ = 2.0**
- System: **Low-pass**, Cutoff frequency: **3 Hz**

**What you see:** The green signal (input) shakes heavily due to the noise.
The orange signal (output) is almost back to a clean sine curve.

**Think about it:** What happens if you lower the cutoff to **0.5 Hz**?
Is the signal filtered even better – or do you lose too much of the original?
            """)

        with st.expander("🟢 Low-pass – Observe the step response"):
            st.markdown("""
**Task:** Watch how quickly the low-pass reacts to a sudden jump.

**Settings:**
- Signal: **Step**, Amplitude **5**, no noise
- System: **Low-pass**, Cutoff frequency: **1 Hz**

**What you see:** The input jumps instantly to 5.
But the output only follows slowly – it "creeps" upward like a heavy object.

**Experiment:** Increase the cutoff to **10 Hz**.
How does the response time change?

→ *Key insight:* The higher the cutoff frequency, the faster the filter can follow.
            """)

        with st.expander("🔵 High-pass – Remove a DC offset"):
            st.markdown("""
**Task:** A signal has an annoying constant offset. Can you filter it out?

**Settings:**
- Signal: **Sine**, Frequency **1 Hz**, Amplitude **3**, DC Offset **+6**
- System: **High-pass**, Cutoff frequency: **0.5 Hz**

**What you see:** The input oscillates around 6 (not 0!).
After the high-pass, the output swings cleanly around zero – the offset is gone.

**Question:** Why can the low-pass filter not do this? Test it!
            """)

        with st.expander("🔵 High-pass – Detect edges in a square wave"):
            st.markdown("""
**Task:** Turn a square wave into a series of short spikes.

**Settings:**
- Signal: **Square**, Frequency **1 Hz**, Amplitude **5**, no offset
- System: **High-pass**, Cutoff frequency: **5 Hz**

**What you see:** The square jumps between +5 and −5.
The high-pass reacts only to these jumps → short needle-like spikes appear.
Afterwards, the signal "creeps" back toward zero.

→ *Real-world use:* This is exactly how edge detection and differentiators work in practice.
            """)

        with st.expander("🟣 Non-linear – Frequency doubling"):
            st.markdown("""
**Task:** Observe that squaring a signal creates a completely new frequency.

**Settings:**
- Signal: **Sine**, Frequency **2 Hz**, Amplitude **3**, Offset **0**
- System: **Non-linear System**

**What you see:** The output oscillates at **4 Hz** –
even though we only fed in 2 Hz!

The squaring system has **created a new frequency** that was not there before.
This is never possible with a low-pass or high-pass filter.

→ *Hint:* In the next lesson you will learn to visualise this using a frequency spectrum.
            """)

        with st.expander("🟣 Non-linear – The rectifier effect"):
            st.markdown("""
**Task:** Turn an alternating signal into a constant one.

**Settings:**
- Signal: **Square**, Frequency **1 Hz**, Amplitude **4**, Offset **0**
- System: **Non-linear System**

**What you see:** The input jumps between +4 and −4.
- (+4)² = **16**
- (−4)² = **16**
→ The output is a flat line at **16** – constant, even though the input keeps changing!

→ *Key insight:* The squaring system always produces a positive number.
It "forgets" the sign of the input.
            """)


if __name__ == "__main__":
    main()
