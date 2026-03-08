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
    """Safely initialize session state variables."""
    if key not in st.session_state:
        st.session_state[key] = default_value

# Strict State Initialization 
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
    """Extremely robust callback for handling JSON config uploads in the Cloud."""
    if st.session_state.json_uploader is not None:
        try:
            data = json.load(st.session_state.json_uploader)
            for k, v in data.items():
                if k in st.session_state:
                    # Enforce type matching to prevent slider crashes in the Cloud
                    if isinstance(st.session_state[k], float):
                        st.session_state[k] = float(v)
                    elif isinstance(st.session_state[k], int):
                        st.session_state[k] = int(v)
                    else:
                        st.session_state[k] = str(v)
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
    # Fixed 3-second simulation 
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

    # 2. Apply System Logic
    y_out = np.zeros_like(y_in)
    
    # Normalize cutoff frequency for digital filters (0 to 1)
    safe_cutoff = min(config.sys_parameter, nyq - 0.1) 
    normal_cutoff = safe_cutoff / nyq

    if "Low-pass" in config.sys_selection:
        # 2nd-Order Butterworth Low-pass (SOS for stability, proper ZI for transient suppression)
        sos = signal.butter(2, normal_cutoff, btype='low', analog=False, output='sos')
        zi = signal.sosfilt_zi(sos) * y_in[0]
        y_out, _ = signal.sosfilt(sos, y_in, zi=zi)
        
    elif "High-pass" in config.sys_selection:
        # 2nd-Order Butterworth High-pass
        sos = signal.butter(2, normal_cutoff, btype='high', analog=False, output='sos')
        zi = signal.sosfilt_zi(sos) * y_in[0]
        y_out, _ = signal.sosfilt(sos, y_in, zi=zi)
        
    elif "Non-linear" in config.sys_selection:
        # System 3: Non-linear squarer (y = x^2)
        y_out = y_in ** 2

    return t, y_in, y_out

# ==========================================
# --- 3. MAIN UI STRUCTURE ---
# ==========================================
def main():
    st.title("🎯 Intuitive System Dynamics")
    st.markdown("Easily learn how different systems shape and change basic signals.")

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
            sig_freq = st.slider("Frequency (Hz)", 0.5, 20.0, step=0.5, key="sig_freq", help="How fast the signal oscillates.")
            sig_amp = st.slider("Amplitude", 1.0, 10.0, step=1.0, key="sig_amp", help="The height of the signal.")
            sig_offset = st.slider("DC Offset", -10.0, 10.0, step=0.5, key="sig_offset", help="Shifts the signal up or down.")
            sig_noise = st.slider("Sensor Noise", 0.0, 5.0, step=0.1, key="sig_noise", help="Adds random fluctuations.")
            
            st.divider()
            
            st.header("2. Choose System")
            sys_options = [
                "System 1: Low-pass Filter", 
                "System 2: High-pass Filter", 
                "System 3: Non-linear (y = x²)"
            ]
            sys_sel = st.selectbox("Target System:", sys_options, key="sys_sel")
            
            # Context-sensitive UI: Hide filter cutoff for non-linear system
            if "Non-linear" not in sys_sel:
                sys_param = st.slider("Cutoff Frequency (Hz)", 0.1, 20.0, step=0.1, key="sys_param")
            else:
                sys_param = st.session_state.sys_param # Keep value in state but don't render slider

            st.divider()
            
            # JSON File Operations
            st.header("💾 Save / Load Config")
            keys_to_save = ['sig_form', 'sig_freq', 'sig_amp', 'sig_offset', 'sig_noise', 'sys_sel', 'sys_param']
            current_conf = {k: st.session_state[k] for k in keys_to_save}
            
            st.download_button(
                label="📥 Export Config (JSON)", 
                data=json.dumps(current_conf, indent=2), 
                file_name="system_config.json", 
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
            form=st.session_state.sig_form, 
            frequency=st.session_state.sig_freq, 
            amplitude=st.session_state.sig_amp, 
            offset=st.session_state.sig_offset, 
            noise_amp=st.session_state.sig_noise,
            sys_selection=st.session_state.sys_sel, 
            sys_parameter=st.session_state.sys_param
        )
        t, y_in, y_out = simulate_system(config)

        # --- VISUAL FEEDBACK METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Max", f"{np.max(y_in):.1f}")
        col2.metric("Output Max", f"{np.max(y_out):.1f}")
        col3.metric("Output Min", f"{np.min(y_out):.1f}")

        st.markdown("---")

        # --- PLOTLY VISUALIZATION ---
        fig = go.Figure()
        
        # Input: Dotted Green
        fig.add_trace(go.Scatter(x=t, y=y_in, mode='lines', name='Input (Original)', line=dict(color='#00CC96', width=2, dash='dot')))
        # Output: Solid Orange
        fig.add_trace(go.Scatter(x=t, y=y_out, mode='lines', name='Output (Processed)', line=dict(color='#FF9F1C', width=4)))

        # Dynamic y-axis scaling to handle y=x^2 properly
        y_max_abs = max(np.max(np.abs(y_in)), np.max(np.abs(y_out)))
        y_range = [-y_max_abs * 1.1, y_max_abs * 1.1]
        
        # For squaring, the output is purely positive, we might need a better view
        if "Non-linear" in config.sys_selection and config.offset == 0:
            y_range = [min(-config.amplitude*1.1, -1), y_max_abs * 1.1]

        fig.update_layout(
            xaxis_title='Time (Seconds)', 
            yaxis_title='Amplitude',
            yaxis_range=y_range,
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
                file_name='system_simulation_data.csv',
                mime='text/csv',
                use_container_width=True
            )

    with tab2:
        st.header("📘 System Theory: Linear vs. Non-linear")
        st.markdown("""
        * **🟢 System 1 (Low-pass Filter):** A 2nd-Order Butterworth filter. It lets slow signals and constant offsets pass but aggressively blocks fast oscillations and noise. It is a *linear* system.
        * **🔵 System 2 (High-pass Filter):** A 2nd-Order Butterworth filter. It blocks slow/constant signals (like a DC Offset) and only lets sudden changes jump through. It is also a *linear* system.
        * **🟣 System 3 (Non-linear squarer):** This system uses the math $y(t) = (x(t))^2$. It is **non-linear**. Unlike filters, which only change the amplitude and phase of existing frequencies, non-linear systems can create entirely *new* frequencies!
        """)

    with tab3:
        st.header("📝 Practical Exercises")
        st.markdown("Test your understanding! Adjust the parameters in **Tab 1** and observe how the output signal changes.")

        with st.expander("🟢 System 1: Low-pass Filter", expanded=True):
            st.markdown("""
            * **Exercise 1 (Filtering Noise):** Set a **Sine** wave at **2.0 Hz** and add **1.5 Sensor Noise**. Drop the Cutoff Frequency to **3.0 Hz**. Notice how the filter magically recovers the clean sine wave from the noise.
            * **Exercise 2 (Square Wave Smoothing):** Set the signal to **Square** at **1.0 Hz** (0 Noise). Set the Cutoff Frequency to **1.0 Hz**. Observe how the harsh corners are rounded off.
            """)

        with st.expander("🔵 System 2: High-pass Filter"):
            st.markdown("""
            * **Exercise 1 (DC Offset Removal):** Set a **Sine** wave at **1.0 Hz** and add a **DC Offset of 5.0**. Now set the Cutoff Frequency to **0.5 Hz**. Observe how the orange output line perfectly ignores the offset and centers back around zero!
            * **Exercise 2 (Edge Detection):** Select a **Square** wave at **1.0 Hz** (0 Offset). Set the Cutoff Frequency to **5.0 Hz**. See how the flat tops drop toward zero, leaving only sharp spikes.
            """)

        with st.expander("🟣 System 3: Non-linear (y = x²)"):
            st.markdown("""
            * **Exercise 1 (Frequency Doubling):** Set a **Sine** wave to **1.0 Hz**, Amplitude **2.0**, Offset **0.0**. Look at the orange line. It is purely positive (because negative x negative = positive) and it oscillates **twice as fast** as the input! Squaring a sine wave creates a new frequency.
            * **Exercise 2 (The DC Converter):** Set a **Square** wave to **1.0 Hz**, Amplitude **3.0**, Offset **0.0**. The input jumps between +3 and -3. Since $(+3)^2 = 9$ and $(-3)^2 = 9$, the output is a perfectly flat DC line at 9.0!
            * **Exercise 3 (Parabolic Shapes):** Change the signal to a **Sawtooth** wave. A sawtooth is a straight line. Squaring a straight line ($x$) gives a parabola ($x^2$). Look at the beautiful curved shapes at the output!
            """)

if __name__ == "__main__":
    main()