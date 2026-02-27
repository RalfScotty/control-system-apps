import streamlit as st
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from dataclasses import dataclass
import pandas as pd

# ==========================================
# --- 0. PAGE CONFIG & STATE MANAGEMENT ---
# ==========================================
st.set_page_config(page_title="Interactive Signal Generator", layout="wide", page_icon="🌊")

# Custom CSS for the "Educational Dashboard" look
st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size: 18px !important; font-weight: bold !important; }
    .stMetric { background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #303030; }
</style>
""", unsafe_allow_html=True)

def init_state(key, default_value):
    """Secure state management to prevent data loss on rerun."""
    if key not in st.session_state:
        st.session_state[key] = default_value

# Initialize default parameters
init_state('sig_form', "Sine")
init_state('sig_type', "Continuous")
init_state('sig_freq', 1.0)
init_state('sig_amp', 5.0)
init_state('sig_offset', 0.0)
init_state('sig_phase', 0.0)
init_state('sig_samples', 200)
init_state('sig_noise', 0.0) # New: Noise initialization

# ==========================================
# --- 1. LOGIC CLASSES & DATA STRUCTURES ---
# ==========================================

@dataclass
class SignalConfig:
    """Encapsulates all signal parameters for type-safe passing."""
    form: str
    type: str
    frequency: float
    amplitude: float
    offset: float
    phase: float
    samples: int
    noise_amp: float # New: Noise parameter

@st.cache_data(show_spinner=False)
def generate_signal(config: SignalConfig):
    """Separates the calculation logic entirely from the UI."""
    # Start, Stop, Step
    t = np.arange(0, 2 * np.pi + 1 / config.samples, 2 * np.pi / config.samples)
    phase_rad = config.phase * 2 * np.pi / 360
    
    # Mathematical calculation of the base signal
    if config.form == "Sine":
        y = config.amplitude * np.sin(config.frequency * t + phase_rad) + config.offset
    elif config.form == "Square":
        y = config.amplitude * signal.square(config.frequency * t + phase_rad) + config.offset
    elif config.form == "Sawtooth":
        y = config.amplitude * signal.sawtooth(config.frequency * t + phase_rad) + config.offset
    else:
        y = np.zeros_like(t)
        
    # Add Gaussian noise
    if config.noise_amp > 0:
        noise = np.random.normal(0, config.noise_amp, len(t))
        y += noise
        
    return t, y

# ==========================================
# --- 2. MAIN UI STRUCTURE ---
# ==========================================
def main():
    st.title("🌊 Interactive Signal Dashboard")
    st.markdown("Explore the properties of periodic signals and sensor noise in real-time.")

    # Guided workflow through tabs
    tab1, tab2 = st.tabs(["1. 🎛️ Signal Generator", "2. 📘 Theory & Export"])

    with tab1:
        # --- SIDEBAR: Inputs ---
        with st.sidebar:
            st.header("⚙️ Signal Parameters")
            
            # Form (now a Dropdown) and Type
            sig_form = st.selectbox("Signal Form:", ["Sine", "Square", "Sawtooth"], key="sig_form")
            sig_type = st.radio("Signal Type:", ["Continuous", "Discrete"], key="sig_type", horizontal=True)
            
            st.divider()
            
            # Signal Characteristics
            st.subheader("Characteristics")
            sig_freq = st.slider("Frequency (Hz)", 0.1, 10.0, step=0.1, key="sig_freq")
            sig_amp = st.slider("Amplitude", 0.1, 10.0, step=0.1, key="sig_amp")
            sig_offset = st.slider("DC Offset", -10.0, 10.0, step=0.5, key="sig_offset")
            sig_phase = st.slider("Phase (°)", -180.0, 180.0, step=5.0, key="sig_phase")
            
            st.divider()
            
            # Simulation Settings
            st.subheader("Simulation")
            sig_samples = st.slider("Samples / Resolution", 10, 300, step=10, key="sig_samples")
            sig_noise = st.slider("Sensor Noise (Std Dev)", 0.0, 5.0, step=0.1, key="sig_noise")

        # --- MAIN AREA: Logic Execution ---
        current_config = SignalConfig(
            form=sig_form, type=sig_type, frequency=sig_freq, 
            amplitude=sig_amp, offset=sig_offset, phase=sig_phase, 
            samples=sig_samples, noise_amp=sig_noise
        )
        
        # Generate data
        t, y = generate_signal(current_config)

        # --- MAIN AREA: Metrics (Visual Feedback) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Value", f"{np.max(y):.2f}", delta="Peak", delta_color="off")
        col2.metric("Min Value", f"{np.min(y):.2f}", delta="Valley", delta_color="off")
        col3.metric("Peak-to-Peak", f"{np.max(y) - np.min(y):.2f}")
        col4.metric("Resolution", f"{len(t)} pts")

        st.markdown("---")

        # --- MAIN AREA: Plotly Visualization ---
        fig = go.Figure()
        
        if current_config.type == "Continuous":
            # Line plot for continuous signals
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Signal', line=dict(color='#00CC96', width=3)))
        else:
            # Stem plot (Lollipop) for discrete signals
            fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name='Samples', marker=dict(color='#EF553B', size=6)))
            for x_val, y_val in zip(t, y):
                fig.add_shape(type="line", x0=x_val, y0=0, x1=x_val, y1=y_val, line=dict(color="#EF553B", width=1))

        # Layout adjustments
        fig.update_layout(
            xaxis_title='Time t [s]', 
            yaxis_title='Amplitude y(t)',
            yaxis_range=[-15, 15],
            template="plotly_dark",
            height=550,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        # Axis styling
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='#333333', 
            zeroline=True, zerolinewidth=2, zerolinecolor='#888888',
            tickvals=[0.0, np.pi, 2*np.pi],
            ticktext=['0', 'π', '2π']
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='#333333',
            zeroline=True, zerolinewidth=2, zerolinecolor='#888888'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("📘 The Mathematics of Signals")
        st.markdown(r"""
        The generated ideal signal is calculated using the general formula for a harmonic oscillation:
        $$ y(t) = A \cdot \sin(2\pi f \cdot t + \varphi) + y_0 $$
        
        * **$A$**: Amplitude (Maximum displacement)
        * **$f$**: Frequency in Hertz (Cycles per second)
        * **$\varphi$**: Phase shift in radians
        * **$y_0$**: DC Offset
        """)
        
        
        
        st.info("💡 **Continuous vs. Discrete:** A continuous signal exists at every point in time. A discrete signal is only defined at specific sampling points, which is how digital microcontrollers (like Arduino or ESP32) read sensor data.")
        
        if sig_noise > 0:
            st.warning("⚠️ **Sensor Noise Active:** You have added Gaussian noise to the signal. This simulates the random fluctuations you would see when measuring a real signal with an ADC (Analog-to-Digital Converter).")

        st.markdown("### 💾 Export Data")
        # CSV Export for Reproducibility
        df = pd.DataFrame({'Time [s]': t, 'Amplitude y(t)': y})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Signal Data as CSV",
            data=csv,
            file_name=f'signal_{current_config.form.lower()}.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()