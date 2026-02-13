import streamlit as st
import numpy as np
import pandas as pd  # Added for CSV handling
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from dataclasses import dataclass, field
from enum import IntEnum
import json 

# ==========================================
# --- CLASS DEFINITIONS ---
# ==========================================

class TuningState(IntEnum):
    """Enumeration to track the current state of the tuning process."""
    READY = 1        # System is idle or waiting to start
    OSCILLATION = 2  # System is actively inducing oscillations

class TuningMode(IntEnum):
    """Defines how the tuner handles the two-phase relay test."""
    PHASE_1_ONLY = 1    # Standard relay test using full output limits
    ALWAYS_PHASE_2 = 2  # Follows Phase 1 with a centered relay test (better bias)
    AUTO = 3            # Switches to Phase 2 only if Phase 1 quality is poor

class OscillationTuning:
    """
    Implements a Relay Auto-Tuning method to determine PID parameters.
    It induces a limit cycle (oscillation) to find the ultimate gain (Ku) 
    and ultimate period (Tu).
    """
    def __init__(self, sampling_time: float, setpoint: float, min_out: float, max_out: float, 
                 num_periods: int = 3, min_osc_amp: float = 1.0, tuning_mode: int = 3, min_quality: float = 95.0):
        # Time and Setpoint configuration
        self.dt = sampling_time
        self.setpoint = setpoint
        self.min_out_limit = min_out
        self.max_out_limit = max_out
        
        # Tuning configuration
        self.state = TuningState.READY
        self.tuning_time = 0.0
        self.min_osc_amp = min_osc_amp    # Hysteresis/minimum amplitude to ignore noise
        self.target_periods = num_periods # Number of oscillations to analyze
        self.tuning_mode = TuningMode(tuning_mode)
        self.min_quality = min_quality    # Symmetry threshold (percentage)
        
        # Phase management
        self.current_phase = 1            # Phase 1: Full range; Phase 2: Centered around average
        self.relay_high = max_out
        self.relay_low = min_out
        
        # Quality and Calculation metrics
        self.quality_area = 0.0           # Measures symmetry of the oscillation
        self.p1_area = None
        self.kp, self.tn, self.tv = 0.0, 0.0, 0.0 # PID Results
        self.tuning_done = False
        self._reset_oscillation_data()

    def _reset_oscillation_data(self):
        """Resets all buffers and counters for a fresh oscillation analysis."""
        self.num_oscillations = 0
        self.first_setpoint_crossing = False 
        # Arrays to store peaks and timings for averaging
        self.max_vals = np.full(self.target_periods + 2, -np.inf)
        self.min_vals = np.full(self.target_periods + 2, np.inf)
        self.max_times = np.zeros(self.target_periods + 2)
        
        # Area calculation for symmetry check (Quality)
        self.out_sum = 0.0
        self.out_count = 0
        self.area_pos = 0.0
        self.area_neg = 0.0
        self.output = 0.0

    def check_quality(self):
        """
        Evaluates the symmetry of the oscillation.
        In a linear system, the area above and below the setpoint should be equal.
        """
        max_area = max(self.area_pos, self.area_neg)
        if max_area > 1e-6:
            # Quality is the ratio between the smaller and larger area
            self.quality_area = (min(self.area_pos, self.area_neg) / max_area) * 100.0
        else:
            self.quality_area = 0.0
        
        finish_tuning = False
        start_phase_2 = False

        # Logic to decide if we need a second phase (centered relay) to improve accuracy
        if self.tuning_mode == TuningMode.PHASE_1_ONLY:
            finish_tuning = True
        elif self.tuning_mode == TuningMode.ALWAYS_PHASE_2:
            if self.current_phase == 1: start_phase_2 = True
            else: finish_tuning = True
        else: # AUTO Mode
            if self.quality_area >= self.min_quality: 
                finish_tuning = True
            elif self.current_phase == 1: 
                start_phase_2 = True
            else: 
                finish_tuning = True

        if finish_tuning:
            self._calculate_parameters()
        elif start_phase_2:
            # Transitioning to Phase 2: Calculate new relay levels based on process bias
            self.p1_area = self.quality_area
            self.current_phase = 2
            # Find the average output required to maintain the setpoint
            avg_out = self.out_sum / (self.out_count if self.out_count > 0 else 1)
            # Symmetrically apply relay steps around that average
            delta_out = min(self.max_out_limit - avg_out, avg_out - self.min_out_limit)
            self.relay_high = avg_out + delta_out
            self.relay_low = avg_out - delta_out
            self._reset_oscillation_data()
            self.state = TuningState.OSCILLATION

    def _calculate_parameters(self):
        """Calculates PID parameters using the Ziegler-Nichols frequency response formula."""
        idx = self.num_oscillations
        # Tu: Ultimate Period (time between peaks)
        tu = (self.max_times[idx] - self.max_times[idx-1]) 
        
        if idx > 0:
            avg_max = np.mean(self.max_vals[1:idx+1])
            avg_min = np.mean(self.min_vals[1:idx+1])
            avg_amp = (avg_max - avg_min) / 2.0 # Process amplitude
        else:
            avg_amp = 1.0
            
        # h: Amplitude of the relay (input)
        h = (self.relay_high - self.relay_low) / 2.0
        # Ku: Ultimate Gain (Describing Function Analysis: Ku = 4*h / (pi*amp))
        ku = (4.0 * h) / (np.pi * avg_amp) if avg_amp > 0 else 0
        
        # Classical Ziegler-Nichols tuning rules for a PID controller
        self.kp, self.tn, self.tv = 0.6 * ku, 0.5 * tu, 0.125 * tu
        self.tuning_done = True
        self.state = TuningState.READY

    def step(self, actual_value: float) -> float:
        """
        Main execution step to be called in every control loop cycle.
        Returns the output for the actuator.
        """
        self.tuning_time += self.dt
        
        # Initialization
        if self.state == TuningState.READY:
            self._reset_oscillation_data()
            self.state = TuningState.OSCILLATION
            # Initial kick: Output high if below setpoint
            self.output = self.relay_high if actual_value <= self.setpoint else self.relay_low

        elif self.state == TuningState.OSCILLATION:
            # Wait for first crossing to avoid counting the initial transient
            if not self.first_setpoint_crossing:
                if (self.output == self.relay_high and actual_value >= self.setpoint) or \
                   (self.output == self.relay_low and actual_value <= self.setpoint):
                    self.first_setpoint_crossing = True
            
            # Record peak (maximum) and valley (minimum) values for the current period
            idx = min(self.num_oscillations, len(self.max_vals)-1)
            if actual_value > self.max_vals[idx]:
                self.max_vals[idx] = actual_value
                self.max_times[idx] = self.tuning_time
            if self.first_setpoint_crossing:
                if actual_value < self.min_vals[idx]:
                    self.min_vals[idx] = actual_value

            # Integrate area for quality/symmetry analysis
            if self.num_oscillations >= 1:
                error = actual_value - self.setpoint
                if error > 0: self.area_pos += error * self.dt
                else: self.area_neg += abs(error) * self.dt

            # Bang-bang control logic (The Relay)
            control_deviation = self.setpoint - actual_value
            if control_deviation > self.min_osc_amp: 
                self.output = self.relay_high
            elif control_deviation < -self.min_osc_amp: 
                self.output = self.relay_low

            # Track average output to find process bias
            if self.num_oscillations >= 1:
                self.out_sum += self.output
                self.out_count += 1

            # Check if a full oscillation period has completed
            if (self.max_vals[idx] > (self.setpoint + self.min_osc_amp) and 
                self.min_vals[idx] < (self.setpoint - self.min_osc_amp)):
                # If we just switched from low to high and crossed the setpoint
                if self.output == self.relay_high and control_deviation > 0: 
                     if self.num_oscillations < self.target_periods: 
                         self.num_oscillations += 1
                     else:
                         # Enough data collected, evaluate quality and finish/switch phase
                         self.check_quality()
                         self.output = 0.0
        return self.output

# ==========================================
# --- STREAMLIT HELPERS & LOGIC ---
# ==========================================

@dataclass
class PIDConfig:
    kp: float = 2.0
    tn: float = 10.0
    tv: float = 1.0
    tf: float = 0.1
    ctrl_type: str = "PID"
    anti_windup: str = "ON"
    d_kickoff: str = "OFF"
    setpoints: np.ndarray = field(default_factory=lambda: np.array([]))

def system_dynamics(x, t, u, gain, time_const, damping, order):
    """Defines the differential equations for PT1, PT2, and PT3 systems."""
    if order == "PT1":
        return [(gain * u - x[0]) / time_const, 0]
    elif order == "PT2":
        y, v = x
        # Standard second-order differential equation
        dvdt = (-2.0 * damping * time_const * v - y + gain * u) / (time_const**2)
        return [v, dvdt]
    elif order == "PT3":
        y, v = x
        # Simplified third-order approximation
        dvdt = (-3.0 * time_const * v - y + gain * u) / (time_const**3) 
        return [v, dvdt]
    return [0, 0]

def simulate_pid_response(time_vector, model_params, pid_conf):
    """Simulates the closed-loop response of the PID controller."""
    ns = len(time_vector) - 1
    dt = time_vector[1] - time_vector[0]
    
    # Effective derivative time considering the filter constant Tf
    tv_eff = pid_conf.tv * (pid_conf.tf / dt * (1.0 - np.exp(-dt / pid_conf.tf))) if pid_conf.tf > 0 else pid_conf.tv
    
    op, e, ie = np.zeros(ns + 1), np.zeros(ns + 1), np.zeros(ns + 1)
    p_term, i_term, d_term = np.zeros(ns + 1), np.zeros(ns + 1), np.zeros(ns + 1)
    pv = np.zeros((ns + 1, 2))
    
    # Dead time (Delay) implementation via ring buffer
    delay_samples = int(max(1, np.ceil(model_params['delay'] / dt))) 
    ring_buffer = np.zeros(delay_samples)
    buffer_idx = 0

    for i in range(ns):
        e[i] = pid_conf.setpoints[i] - pv[i, 0]
        
        # Proportional Term
        if pid_conf.ctrl_type in ["P", "PI", "PD", "PID"]: 
            p_term[i] = pid_conf.kp * e[i]
        
        # Integral Term
        if pid_conf.ctrl_type in ["PI", "PID"]:
            prev_ie = ie[i-1] if i >= 1 else 0.0
            ie[i] = prev_ie + e[i] * dt
            i_term[i] = (pid_conf.kp / pid_conf.tn) * ie[i] if pid_conf.tn > 0 else 0
        
        # Derivative Term (with filtering)
        if pid_conf.ctrl_type in ["PD", "PID"] and i >= 1:
            # D-Kickoff: use error change vs PV change to avoid spikes on setpoint steps
            delta_input = (e[i] - e[i-1]) if pid_conf.d_kickoff == "ON" else (-pv[i, 0] + pv[i-1, 0])
            decay = np.exp(-dt / pid_conf.tf)
            d_term[i] = (pid_conf.kp * tv_eff / pid_conf.tf) * delta_input + d_term[i-1] * decay

        raw_op = p_term[i] + i_term[i] + d_term[i]
        op[i] = np.clip(raw_op, -100.0, 100.0) # Output saturation
        
        # Anti-Windup Logic
        if pid_conf.anti_windup == "ON":
            is_sat = (raw_op > 100.0 and e[i] > 0) or (raw_op < -100.0 and e[i] < 0)
            if is_sat:
                ie[i] -= e[i] * dt # Stop integration during saturation
                i_term[i] = (pid_conf.kp / pid_conf.tn) * ie[i] if pid_conf.tn > 0 else 0
        
        # Apply Delay
        delayed_u = ring_buffer[buffer_idx]
        ring_buffer[buffer_idx] = op[i]
        buffer_idx = (buffer_idx + 1) % delay_samples
        
        # Solve ODE for the next time step
        y_next = odeint(system_dynamics, pv[i], [0, dt], 
                        args=(delayed_u, model_params['gain'], model_params['time_const'], 
                              model_params['damping'], model_params['order']))
        
        # Add Measurement Noise
        noise_amp = model_params['noise']
        noise = np.random.normal(0, noise_amp) if noise_amp > 0 else 0.0
        
        pv[i+1, 0] = y_next[-1, 0] + noise
        pv[i+1, 1] = y_next[-1, 1]
        
    return pv, op, e, p_term, i_term, d_term

# --------------------------------------------------------------------------------
# CALLBACK: Load Configuration
# Runs BEFORE the script re-runs to prevent "widget already instantiated" errors.
# --------------------------------------------------------------------------------
def load_config_callback():
    if st.session_state.uploader_key is not None:
        try:
            # Read file from uploader (available in session state via key)
            uploaded_file = st.session_state.uploader_key
            data = json.load(uploaded_file)
            
            # Update session state with loaded values
            for k, v in data.items():
                st.session_state[k] = v
            
            st.toast("✅ Configuration loaded successfully!", icon="💾")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# ==========================================
# --- MAIN UI STRUCTURE ---
# ==========================================

def main():
    st.set_page_config(page_title="PID Tuner Ultimate", layout="wide")
    
    # CSS styling for bigger, bolder tabs
    st.markdown("""
    <style>
        /* Increase font size and make tabs bold */
        button[data-baseweb="tab"] {
            font-size: 24px !important;
            font-weight: bold !important;
        }
        /* Fallback for Markdown inside Tab buttons */
        button[data-baseweb="tab"] div p {
            font-size: 24px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🕹️ Control Engineering Dashboard")

    # Session State Initialization
    if 'kp' not in st.session_state: st.session_state.kp = 2.0
    if 'tn' not in st.session_state: st.session_state.tn = 10.0
    if 'tv' not in st.session_state: st.session_state.tv = 1.0

    # Define Tabs
    tab1, tab2, tab3 = st.tabs([
        "1. 🏗️ Plant Model", 
        "2. 📈 Oscillation Tuning", 
        "3. 🚀 PID Test"
    ])

    # ----------------------------------------------------
    # TAB 1: MODEL
    # ----------------------------------------------------
    with tab1:
        with st.sidebar:
            st.header("1. Plant Parameters")
            
            m_order = st.selectbox("System Order", ["PT1", "PT2", "PT3"], index=1, key="m_order")
            m_v = st.slider("Gain (K)", 0.1, 10.0, 2.8, 0.1, key="m_v")
            m_t = st.slider("Time Constant (T)", 0.1, 20.0, 2.0, 0.1, key="m_t")
            
            if m_order == "PT2":
                m_xi = st.slider("Damping Ratio (D)", 0.1, 3.0, 1.5, key="m_xi")
            else:
                m_xi = 1.0
                st.session_state.m_xi = 1.0
            
            m_delay = st.slider("Dead Time [s]", 0.0, 10.0, 2.0, key="m_delay")
            st.divider()
            m_noise_amp = st.slider("Measurement Noise (Amp)", 0.0, 1.0, 0.0, 0.01, key="m_noise_amp")

        model_params = {
            "order": m_order, "gain": m_v, "time_const": m_t,
            "damping": m_xi, "delay": m_delay, "noise": m_noise_amp
        }

        st.markdown(f"### Model Preview: {m_order} with {m_delay}s Dead Time")
        
        # Calculate Open Loop Step Response
        t_open = np.linspace(0, 200 + m_delay, 1000)
        dt_o = t_open[1] - t_open[0]
        nd_o = int(max(1, np.ceil(m_delay / dt_o)))
        rb_o = np.zeros(nd_o); idx_o = 0
        pv_o = []; curr_o = [0, 0]
        
        for _ in t_open:
            u_del = rb_o[idx_o]
            rb_o[idx_o] = 1.0 # Unit Step
            idx_o = (idx_o + 1) % nd_o
            step_o = odeint(system_dynamics, curr_o, [0, dt_o], args=(u_del, m_v, m_t, m_xi, m_order))
            
            # Add Noise
            noise = np.random.normal(0, m_noise_amp) if m_noise_amp > 0 else 0.0
            pv_o.append(step_o[-1, 0] + noise)
            
            curr_o = step_o[-1]
        
        # Store Model Data for CSV Export
        df_model = pd.DataFrame({
            "Time": t_open,
            "Step_Response": pv_o
        })
        st.session_state['data_model'] = df_model

        fig_open = go.Figure()
        fig_open.add_trace(go.Scatter(x=t_open, y=pv_o, name="Step Response", line=dict(color='#00CC96', width=2)))
        fig_open.update_layout(
            title="Open Loop Step Response (with Noise Preview)", 
            xaxis_title="Time [s]", yaxis_title="Output y(t)",
            template="plotly_dark", height=600
        )
        st.plotly_chart(fig_open, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: TUNING
    # ----------------------------------------------------
    with tab2:
        with st.sidebar:
            st.header("2. Autotuning Setup")
            
            t_mode_sel = st.selectbox("Algorithm", options=[1, 2, 3], index=2,
                                      format_func=lambda x: {
                                          1: "Phase 1 Only (Fast)", 
                                          2: "Symmetric (Phase 1+2)", 
                                          3: "Automatic (Smart)"
                                      }[x], key="t_mode_sel")
            
            if t_mode_sel == 3:
                t_min_qual = st.slider("Min. Symmetry Quality [%]", 50, 100, 95, key="t_min_qual")
            else:
                t_min_qual = 95.0

            st.divider()
            st.markdown("**Relay Parameters**")
            t_set = st.number_input("Setpoint (Operating Point)", value=150.0, key="t_set")
            t_hyst = st.number_input("Hysteresis", value=1.0, key="t_hyst")
            t_max = st.number_input("Relay Max", value=100.0, key="t_max")
            t_min = st.number_input("Relay Min", value=-100.0, key="t_min")
            t_periods = st.slider("Required Periods", 2, 10, 4, key="t_periods")

        st.subheader("Autotuning Simulation")
        start_tuning = st.button("▶️ Start Tuning", type="primary")

        if start_tuning:
            Ts = 0.05
            t_sim = np.arange(0, 1000, Ts) 
            
            tuner = OscillationTuning(
                sampling_time=Ts, setpoint=t_set, min_out=t_min, max_out=t_max, 
                num_periods=t_periods, min_osc_amp=t_hyst,
                tuning_mode=t_mode_sel, min_quality=t_min_qual
            )
            
            y_hist, u_hist = [], []
            curr_state = np.zeros(2)
            y_prev = 0.0
            
            nd_t = int(max(1, np.ceil(m_delay / Ts)))
            rb_t = np.zeros(nd_t); idx_t = 0
            
            progress_bar = st.progress(0)
            for i, _ in enumerate(t_sim):
                noise_amp = model_params['noise']
                noise_val = np.random.normal(0, noise_amp) if noise_amp > 0 else 0.0
                y_measured = y_prev + noise_val
                
                u_cmd = tuner.step(y_measured)
                
                u_delayed = rb_t[idx_t]
                rb_t[idx_t] = u_cmd
                idx_t = (idx_t + 1) % nd_t

                y_step = odeint(system_dynamics, curr_state, [0, Ts], 
                                args=(u_delayed, m_v, m_t, m_xi, m_order))
                
                y_hist.append(y_measured); u_hist.append(u_cmd)
                y_prev = y_step[-1, 0]; curr_state = y_step[-1]
                
                if i % 50 == 0: progress_bar.progress(min(i / len(t_sim), 1.0))
                if tuner.tuning_done: break
            
            progress_bar.progress(100)
            
            # Store Tuning Data for CSV Export
            t_axis = np.arange(len(y_hist)) * Ts
            df_tuning = pd.DataFrame({
                "Time": t_axis,
                "Process_Value": y_hist,
                "Setpoint": [t_set]*len(y_hist),
                "Output": u_hist
            })
            st.session_state['data_tuning'] = df_tuning

            # Store results in session state
            st.session_state.kp = tuner.kp
            st.session_state.tn = tuner.tn
            st.session_state.tv = tuner.tv
            
            # Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Recommended Kp", f"{tuner.kp:.3f}")
            c2.metric("Recommended Tn", f"{tuner.tn:.3f}s")
            c3.metric("Recommended Tv", f"{tuner.tv:.3f}s")
            
            if tuner.p1_area is not None:
                p1_val = tuner.p1_area
                p1_delta = "Good" if p1_val > 90 else "Low"
                c4.metric("Phase 1 Quality", f"{p1_val:.1f}%", delta=p1_delta)
            else:
                c4.metric("Phase 1 Quality", "N/A")

            final_qual = tuner.quality_area
            final_delta = "Good" if final_qual > 90 else "Low"
            c5.metric("Phase 2 Quality", f"{final_qual:.1f}%", delta=final_delta)

            # Plot
            fig_t = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_t.add_trace(go.Scatter(x=t_axis, y=y_hist, name="Process Value (PV)", line=dict(color='cyan')), row=1, col=1)
            fig_t.add_trace(go.Scatter(x=t_axis, y=[t_set]*len(t_axis), name="Setpoint (SP)", line=dict(dash='dash', color='red')), row=1, col=1)
            fig_t.add_trace(go.Scatter(x=t_axis, y=u_hist, name="Relay (OP)", line=dict(shape='hv', color='orange')), row=2, col=1)
            fig_t.update_layout(height=600, template="plotly_dark", title="Tuning Progress")
            st.plotly_chart(fig_t, use_container_width=True)
        
    # ----------------------------------------------------
    # TAB 3: PID TEST
    # ----------------------------------------------------
    with tab3:
        with st.sidebar:
            st.header("3. Controller Test")
            
            use_tuned = st.checkbox("Use Tuned Parameters", value=True, key="use_tuned")
            
            st.markdown("---")
            st.markdown("**PID Parameters**")
            
            def_kp = st.session_state.kp if use_tuned else 1.0
            def_tn = st.session_state.tn if use_tuned else 5.0
            def_tv = st.session_state.tv if use_tuned else 0.0

            rk_type = st.selectbox("Controller Type", ["PID", "PI", "P", "PD"], index=0, key="rk_type")
            
            rk_p = st.number_input("Kp (Gain)", value=def_kp, format="%.4f", key="rk_p")
            rk_n = st.number_input("Tn (Reset Time)", value=def_tn, format="%.4f", key="rk_n")
            rk_v = st.number_input("Tv (Derivative Time)", value=def_tv, format="%.4f", key="rk_v")
            
            rk_tf = rk_v / 10.0 if rk_v > 0 else 0.0
            
            st.markdown("---")
            st.markdown("**Simulation Settings**")
            rk_aw = "ON"
            m_sim_time = st.slider("Duration [s]", 10, 500, 200, key="m_sim_time")
            step_val = st.number_input("Setpoint Step Value", value=st.session_state.get('t_set', 150.0), key="step_val")
            
        st.subheader("Closed-Loop Control Performance")
        start_sim = st.button("🚀 Simulate Step Response", type="primary")

        if start_sim:
            t_p = np.linspace(0, m_sim_time, int(m_sim_time * 20))
            sp_vector = np.ones(len(t_p)) * step_val
            
            pid_conf = PIDConfig(
                kp=rk_p, tn=rk_n, tv=rk_v, tf=rk_tf,
                ctrl_type=rk_type, anti_windup=rk_aw, d_kickoff="OFF",
                setpoints=sp_vector
            )
            
            pv, op, _, _, _, _ = simulate_pid_response(t_p, model_params, pid_conf)
            
            # Store PID Data for CSV Export
            df_pid = pd.DataFrame({
                "Time": t_p,
                "Process_Value": pv[:,0],
                "Setpoint": sp_vector,
                "Output": op
            })
            st.session_state['data_pid'] = df_pid

            # Analysis
            max_pv = np.max(pv[:, 0])
            overshoot = (max_pv - step_val) / step_val * 100 if max_pv > step_val else 0.0
            
            res1, res2, res3 = st.columns(3)
            res1.metric("Max. Overshoot", f"{overshoot:.2f}%", delta_color="inverse")
            res2.metric("Steady-State Value", f"{pv[-1, 0]:.2f}")
            res3.metric("Control Deviation (Static)", f"{step_val - pv[-1, 0]:.4f}")

            # Plot
            fig_pid = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=("Process Value (PV) vs Setpoint (SP)", "Output Power (OP)"),
                                    row_heights=[0.6, 0.4])
            
            fig_pid.add_trace(go.Scatter(x=t_p, y=pv[:,0], name="Actual (PV)", line=dict(color='#00CC96', width=2)), row=1, col=1)
            fig_pid.add_trace(go.Scatter(x=t_p, y=sp_vector, name="Target (SP)", line=dict(dash='dash', color='red')), row=1, col=1)
            fig_pid.add_trace(go.Scatter(x=t_p, y=op, name="Controller Output (OP)", fill='tozeroy', line=dict(color='gray')), row=2, col=1)
            
            fig_pid.update_layout(height=700, template="plotly_dark", xaxis2_title="Time [s]")
            st.plotly_chart(fig_pid, use_container_width=True)

    # ----------------------------------------------------
    # NEW FEATURE: GLOBAL CSV EXPORT
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Data (CSV)")

    # Check if we have any data to export
    has_model_data = 'data_model' in st.session_state
    has_tuning_data = 'data_tuning' in st.session_state
    has_pid_data = 'data_pid' in st.session_state

    if not (has_model_data or has_tuning_data or has_pid_data):
        st.info("Run a simulation (Model, Tuning, or PID Test) to enable CSV export.")
    else:
        # Create a combined dataframe for download
        # Since time scales differ, we concat them side-by-side with prefixes
        frames = []
        
        if has_model_data:
            df_m = st.session_state['data_model'].copy()
            df_m.columns = ["Model_" + c for c in df_m.columns]
            frames.append(df_m)
            
        if has_tuning_data:
            df_t = st.session_state['data_tuning'].copy()
            df_t.columns = ["Tuning_" + c for c in df_t.columns]
            frames.append(df_t)
            
        if has_pid_data:
            df_p = st.session_state['data_pid'].copy()
            df_p.columns = ["PID_" + c for c in df_p.columns]
            frames.append(df_p)
        
        # Concat along axis 1 (side by side). Missing rows will be NaN.
        master_df = pd.concat(frames, axis=1)
        
        csv_data = master_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download All Plot Data (.csv)",
            data=csv_data,
            file_name="sim_results_master.csv",
            mime="text/csv"
        )

    # ----------------------------------------------------
    # SIDEBAR SAVE/LOAD
    # ----------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.header("💾 File Operations")
        
        keys_to_save = [
            "m_order", "m_v", "m_t", "m_xi", "m_delay", "m_noise_amp", 
            "t_mode_sel", "t_min_qual", "t_set", "t_hyst", "t_max", "t_min", "t_periods", 
            "use_tuned", "rk_type", "rk_p", "rk_n", "rk_v", "m_sim_time", "step_val", 
            "kp", "tn", "tv" 
        ]
        
        # Save Configuration
        current_config = {k: st.session_state[k] for k in keys_to_save if k in st.session_state}
        json_string = json.dumps(current_config, indent=2)
        
        st.download_button(
            label="📥 Save Configuration (JSON)",
            data=json_string,
            file_name="pid_config.json",
            mime="application/json"
        )
        
        # Load Configuration
        st.file_uploader("📤 Load Configuration", type=["json"], 
                         key="uploader_key", 
                         on_change=load_config_callback)

if __name__ == "__main__":
    main()