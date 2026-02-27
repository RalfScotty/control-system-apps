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
st.set_page_config(page_title="Intuitive Systems & Signals", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size: 18px !important; font-weight: bold !important; }
    .stMetric { background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #303030; }
</style>
""", unsafe_allow_html=True)

def init_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

# State Initialization
init_state('sig_form', "Sine")
init_state('sig_freq', 1.0)
init_state('sig_amp', 5.0)
init_state('sig_offset', 0.0)
init_state('sig_noise', 0.0)
init_state('sys_sel', "System 1: Low-pass Filter")
init_state('sys_param', 1.0) 

# ==========================================
# --- 1. CALLBACKS ---
# ==========================================
def on_upload_callback():
    if st.session_state.json_uploader is not None:
        try:
            data = json.load(st.session_state.json_uploader)
            for k, v in data.items():
                if k in st.session_state:
                    st.session_state[k] = v
            st.toast("✅ Configuration loaded successfully!", icon="💾")
        except Exception as e:
            st.error(f"Error loading JSON file: {e}")

# ==========================================
# --- 2. LOGIC CLASSES & DATA STRUCTURES ---
# ==========================================
@dataclass
class SimConfig:
    form: str
    frequency: float
    amplitude: float
    offset: float
    noise_amp: float
    sys_selection: str
    sys_parameter: float

@st.cache_data(show_spinner=False)
def simulate_system(config: SimConfig):
    # Fixed 3-second simulation for intuitive understanding of Hz
    dt = 0.001 
    fs = 1.0 / dt
    nyq = 0.5 * fs
    t = np.arange(0, 3.0, dt) 
    
    # 1. Generate Basic Signal with Offset
    if config.form == "Sine":
        y_in = config.amplitude * np.sin(2 * np.pi * config.frequency * t) + config.offset
    elif config.form == "Square":
        y_in = config.amplitude * signal.square(2 * np.pi * config.frequency * t) + config.offset
    elif config.form == "Sawtooth":
        y_in = config.amplitude * signal.sawtooth(2 * np.pi * config.frequency * t) + config.offset
    else:
        y_in = np.zeros_like(t) + config.offset

    # Add Gaussian Noise
    if config.noise_amp > 0:
        noise = np.random.normal(0, config.noise_amp, len(t))
        y_in += noise

    # 2. Apply 2nd-Order Butterworth System Logic
    y_out = np.zeros_like(y_in)
    
    # Normalize cutoff frequency for digital filters (0 to 1)
    safe_cutoff = min(config.sys_parameter, nyq - 0.1) 
    normal_cutoff = safe_cutoff / nyq

    if "Low-pass" in config.sys_selection:
        # Digital 2nd-Order Butterworth Low-pass
        b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
        y_out = signal.lfilter(b, a, y_in)
        
    elif "High-pass" in config.sys_selection:
        # Digital 2nd-Order Butterworth High-pass
        b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
        y_out = signal.lfilter(b, a, y_in)
        
    elif "All-pass" in config.sys_selection:
        # Analog prototype for 2nd-order Butterworth All-pass, then discretized
        omega_c = 2 * np.pi * config.sys_parameter
        num = [1, -np.sqrt(2)*omega_c, omega_c**2]
        den = [1, np.sqrt(2)*omega_c, omega_c**2]
        d_num, d_den, _ = signal.cont2discrete((num, den), dt, method='bilinear')
        y_out = signal.lfilter(d_num[0], d_den[0], y_in)

    return t, y_in, y_out

# ==========================================
# --- 3. MAIN UI STRUCTURE ---
# ==========================================
def main():
    st.title("🎯 Intuitive System Dynamics (2nd-Order Butterworth)")
    st.markdown("Easily learn how different filters shape and change basic signals.")

    tab1, tab2, tab3 = st.tabs([
        "1. 🚀 Interactive Simulation", 
        "2. 📘 Quick Theory", 
        "3. 📝 Exercises"
    ])

    with tab1:
        # --- SIDEBAR ---
        with st.sidebar:
            st.header("1. Choose Signal")
            sig_form = st.selectbox("Signal Shape:", ["Sine", "Square", "Sawtooth"], key="sig_form")
            sig_freq = st.slider("Frequency (Hz)", 0.5, 100.0, step=0.5, key="sig_freq", help="How fast the signal oscillates.")
            sig_amp = st.slider("Amplitude", 1.0, 10.0, step=1.0, key="sig_amp", help="The height of the signal.")
            sig_offset = st.slider("DC Offset", -10.0, 10.0, step=0.5, key="sig_offset", help="Shifts the signal up or down.")
            sig_noise = st.slider("Sensor Noise", 0.0, 5.0, step=0.1, key="sig_noise", help="Adds random fluctuations.")
            
            st.divider()
            
            st.header("2. Choose System")
            sys_options = [
                "System 1:", 
                "System 2:", 
                "System 3:"
            ]
            sys_sel = st.selectbox("Target System:", sys_options, key="sys_sel")
            
            sys_param = st.slider("Cutoff Frequency (Hz)", 0.1, 20.0, step=0.1, key="sys_param")

            st.divider()
            
            # JSON File Operations
            st.header("💾 Save / Load Config")
            keys_to_save = ['sig_form', 'sig_freq', 'sig_amp', 'sig_offset', 'sig_noise', 'sys_sel', 'sys_param']
            current_conf = {k: st.session_state[k] for k in keys_to_save}
            
            st.download_button(
                label="📥 Export Config (JSON)", 
                data=json.dumps(current_conf, indent=2), 
                file_name="filter_system_config.json", 
                mime="application/json",
                use_container_width=True
            )
            
            st.file_uploader(
                "📤 Import Config (JSON)", 
                type=["json"], 
                key="json_uploader", 
                on_change=on_upload_callback
            )

        # --- LOGIC EXECUTION ---
        config = SimConfig(
            form=sig_form, frequency=sig_freq, amplitude=sig_amp, 
            offset=sig_offset, noise_amp=sig_noise,
            sys_selection=sys_sel, sys_parameter=sys_param
        )
        t, y_in, y_out = simulate_system(config)

        # --- VISUAL FEEDBACK METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Peak (Max)", f"{np.max(y_in):.1f}")
        col2.metric("Output Peak (Max)", f"{np.max(y_out):.1f}")
        
        # Calculate visual attenuation (approximate using max absolute values minus offset)
        in_amp = np.max(np.abs(y_in - config.offset)) if config.amplitude > 0 else 1
        out_amp = np.max(np.abs(y_out - (config.offset if "Low-pass" in config.sys_selection else 0)))
        attenuation_percent = (1 - (out_amp / in_amp)) * 100
        col3.metric("Signal Attenuation", f"{max(0, min(100, attenuation_percent)):.0f}%")

        st.markdown("---")

        # --- PLOTLY VISUALIZATION ---
        fig = go.Figure()
        
        # Input: Dotted Green
        fig.add_trace(go.Scatter(x=t, y=y_in, mode='lines', name='Input (Original)', line=dict(color='#00CC96', width=2, dash='dot')))
        # Output: Solid Orange
        fig.add_trace(go.Scatter(x=t, y=y_out, mode='lines', name='Output (Processed)', line=dict(color='#FF9F1C', width=4)))

        fig.update_layout(
            xaxis_title='Time (Seconds)', 
            yaxis_title='Amplitude',
            yaxis_range=[-15, 15],
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=True, zerolinewidth=2, zerolinecolor='#888888')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=True, zerolinewidth=2, zerolinecolor='#888888')

        st.plotly_chart(fig, use_container_width=True)
        
        # --- CSV DOWNLOAD ---
        df = pd.DataFrame({'Time [s]': t, 'Input': y_in, 'Output': y_out})
        csv = df.to_csv(index=False).encode('utf-8')
        
        col_empty, col_btn = st.columns([3, 1])
        with col_btn:
            st.download_button(
                label="📊 Download CSV Data",
                data=csv,
                file_name='filter_simulation_data.csv',
                mime='text/csv',
                use_container_width=True
            )

    with tab2:
        st.header("📘 What happens here?")
        st.markdown("""
        All systems in this dashboard are designed as **2nd-Order Butterworth Filters**. This means their frequency response is mathematically optimized to be as flat as possible (no unwanted ripple effects), making them the gold standard in signal processing.
        
        * **🟢 System 1 (Low-pass Filter):** The "Smoother". It lets slow signals and constant offsets pass but aggressively blocks fast oscillations and noise. Since it's a 2nd-order filter, it cuts off high frequencies much steeper than a standard 1st-order RC filter.
        * **🔵 System 2 (High-pass Filter):** The "Edge Detector". It blocks slow/constant signals (like DC Offset) and only lets sudden changes (the sharp edges) jump through. 
        * **🟣 System 3 (All-pass Filter):** The "Shifter". The output peak is exactly as high as the input peak. However, different frequencies are shifted (delayed) by different amounts, which completely changes the *shape* of complex waves like squares or sawtooths.
        """)

    with tab3:
        st.header("📝 Practical Exercises")
        st.markdown("Test your understanding! Adjust the parameters in **Tab 1** and observe how the output signal changes.")

        with st.expander("🟢 System 1: Low-pass Filter", expanded=True):
            st.markdown("""
            * **Exercise 1 (Filtering Sensor Noise):** Set a **Sine** wave at **1.0 Hz** and add **1.5 Sensor Noise**. It looks chaotic! Now, drop the Cutoff Frequency to **2.0 Hz**. Notice how the filter magically recovers the clean sine wave from the noise.
            * **Exercise 2 (Square Wave Smoothing):** Set the signal to **Square** at **1.0 Hz** (0 Noise). Set the Cutoff Frequency to **1.0 Hz**. Observe how the harsh corners are rounded off, transforming the rigid square into a smooth, sine-like wave.
            * **Exercise 3 (High-Frequency Rejection):** Set a clean **Sine** wave to a high frequency (**5.0 Hz**). Drop the Cutoff Frequency to **1.0 Hz**. Notice how the orange output amplitude is almost completely flattened. The fast signal is effectively blocked!
            """)

        with st.expander("🔵 System 2: High-pass Filter"):
            st.markdown("""
            * **Exercise 1 (DC Offset Removal):** Set a **Sine** wave at **1.0 Hz** and add a **DC Offset of 5.0**. The green line shifts up. Now set the Cutoff Frequency to **0.5 Hz**. Observe how the orange output line perfectly ignores the offset and centers back around zero!
            * **Exercise 2 (Edge Detection):** Select a **Square** wave at **1.0 Hz** (0 Offset). Set the Cutoff Frequency to **2.0 Hz**. See how the flat tops of the square wave immediately drop toward zero, leaving only sharp "spikes" exactly where the signal jumps.
            * **Exercise 3 (Low-Frequency Blocking):** Set a slow **Sine** wave (**0.5 Hz**). Set the Cutoff Frequency to **5.0 Hz**. The output will be practically a flat line. The slow signal is blocked from passing.
            """)

        with st.expander("🟣 System 3: All-pass Filter"):
            st.markdown("""
            * **Exercise 1 (The Amplitude Check):** Set a **Sine** wave to **1.0 Hz**. Slowly drag the Cutoff Frequency slider from **0.1 Hz to 10.0 Hz**. Look at the "Signal Attenuation" metric. Notice how the amplitude *never* drops.
            * **Exercise 2 (Phase Shift):** Keep the **Sine** wave at **1.0 Hz**. Set the Cutoff Frequency to exactly **1.0 Hz**. The output is shifted in time without losing its strength.
            * **Exercise 3 (Sawtooth Distortion):** Change the signal to **Sawtooth** (**1.0 Hz**). Set the Cutoff Frequency to **2.0 Hz**. Even though no energy is blocked, the orange output shape looks wildly distorted! This happens because a sawtooth is made of many frequencies, and the All-pass delays each frequency by a *different* amount, ruining the original shape.
            """)

if __name__ == "__main__":
    main()