import streamlit as st
import numpy as np
import pandas as pd
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
    Implements a Relay Auto-Tuning method (Åström & Hägglund) to determine PID parameters.
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

def system_dynamics_generic(x, t, u, gain, time_constants, has_integrator):
    """
    Defines differential equations for a cascade of PT1 systems with optional Integrator.
    G(s) = (K / s) * (1/(1+T1s)) * (1/(1+T2s))...  if Integrator is ON
    G(s) = K * (1/(1+T1s)) * (1/(1+T2s))...        if Integrator is OFF
    """
    derivatives = []
    
    current_input = u
    state_idx = 0
    
    # --- Stage 0: Integrator (Optional) ---
    if has_integrator:
        # dx/dt = u
        derivatives.append(current_input)
        current_input = x[state_idx] # Output of integrator is input to lags
        state_idx += 1
    else:
        # If no integrator, apply gain directly to input of first lag
        current_input = u * gain

    # --- Lag Stages (PT1s) ---
    for i, T in enumerate(time_constants):
        y = x[state_idx]
        
        # Apply Gain to the first lag if there was an integrator before it
        # (Standard convention: Integrator is usually 1/s, Gain K is applied to system)
        input_val = current_input * gain if (i == 0 and has_integrator) else current_input
        
        dydt = (input_val - y) / T
        derivatives.append(dydt)
        
        current_input = y # Output of this lag is input to next
        state_idx += 1
        
    return derivatives

def simulate_pid_response(time_vector, model_params, pid_conf):
    """Simulates the closed-loop response of the PID controller."""
    ns = len(time_vector) - 1
    dt = time_vector[1] - time_vector[0]
    
    # Effective derivative time considering the filter constant Tf
    tv_eff = pid_conf.tv * (pid_conf.tf / dt * (1.0 - np.exp(-dt / pid_conf.tf))) if pid_conf.tf > 0 else pid_conf.tv
    
    op, e, ie = np.zeros(ns + 1), np.zeros(ns + 1), np.zeros(ns + 1)
    p_term, i_term, d_term = np.zeros(ns + 1), np.zeros(ns + 1), np.zeros(ns + 1)
    
    # Determine Model Structure for ODE
    time_constants = model_params['time_constants']
    has_integrator = model_params['integrator']
    num_states = len(time_constants) + (1 if has_integrator else 0)
    
    # PV array needs to store current value (output)
    pv_output = np.zeros(ns + 1)
    curr_state = np.zeros(num_states)
    
    # Dead time (Delay) implementation via ring buffer
    delay_samples = int(max(1, np.ceil(model_params['delay'] / dt))) 
    ring_buffer = np.zeros(delay_samples)
    buffer_idx = 0

    for i in range(ns):
        pv_val = curr_state[-1] if num_states > 0 else 0.0 # Output is the last state
        
        # Add noise to measurement (feedback)
        noise_amp = model_params['noise']
        noise = np.random.normal(0, noise_amp) if noise_amp > 0 else 0.0
        pv_measured = pv_val + noise
        
        pv_output[i] = pv_measured 

        e[i] = pid_conf.setpoints[i] - pv_measured
        
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
            delta_input = (e[i] - e[i-1]) if pid_conf.d_kickoff == "ON" else (-pv_measured + (pv_output[i-1] if i>0 else 0))
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
        step_res = odeint(system_dynamics_generic, curr_state, [0, dt], 
                        args=(delayed_u, model_params['gain'], time_constants, has_integrator))
        
        curr_state = step_res[-1]
        
    # Store last point
    pv_output[-1] = curr_state[-1] if num_states > 0 else 0.0
        
    return pv_output, op, e, p_term, i_term, d_term

# --------------------------------------------------------------------------------
# CALLBACK: Load Configuration
# --------------------------------------------------------------------------------
def load_config_callback():
    if st.session_state.uploader_key is not None:
        try:
            uploaded_file = st.session_state.uploader_key
            data = json.load(uploaded_file)
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
    
    st.markdown("""
    <style>
        button[data-baseweb="tab"] { font-size: 24px !important; font-weight: bold !important; }
        button[data-baseweb="tab"] div p { font-size: 24px !important; font-weight: bold !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🕹️ Control Engineering Dashboard")

    if 'kp' not in st.session_state: st.session_state.kp = 2.0
    if 'tn' not in st.session_state: st.session_state.tn = 10.0
    if 'tv' not in st.session_state: st.session_state.tv = 1.0

    # ----------------------------------------------------
    # TABS DEFINITION
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 🏗️ Plant Model", 
        "2. 📈 Oscillation Tuning", 
        "3. 🚀 PID Test",
        "4. 📘 Help & Theory"
    ])

    # ----------------------------------------------------
    # TAB 1: MODEL
    # ----------------------------------------------------
    with tab1:
        with st.sidebar:
            st.header("1. Plant Parameters")
            
            m_order = st.selectbox("System Structure", ["1st Order System", "2nd Order System", "3rd Order System"], index=0, key="m_order")
            
            m_v = st.slider("Gain (K)", 0.1, 10.0, 2.0, 0.1, key="m_v")
            m_integrator = False
            
            # Dynamic Sliders based on Order
            time_consts = []
            
            m_t1 = st.slider("Time Constant 1 (T1)", 0.1, 20.0, 2.0, 0.1, key="m_t1")
            time_consts.append(m_t1)
            
            # Integrator only available for 1st order
            if m_order == "1st Order System":
                m_integrator = st.checkbox("Set Integrator", value=False, key="m_integrator")
            
            if m_order in ["2nd Order System", "3rd Order System"]:
                m_t2 = st.slider("Time Constant 2 (T2)", 0.1, 20.0, 1.0, 0.1, key="m_t2")
                time_consts.append(m_t2)
                
            if m_order == "3rd Order System":
                m_t3 = st.slider("Time Constant 3 (T3)", 0.1, 20.0, 0.5, 0.1, key="m_t3")
                time_consts.append(m_t3)
            
            m_delay = st.slider("Dead Time [s]", 0.0, 10.0, 2.0, key="m_delay")
            st.divider()
            m_noise_amp = st.slider("Measurement Noise (Amp)", 0.0, 1.0, 0.0, 0.01, key="m_noise_amp")

        model_params = {
            "gain": m_v, "time_constants": time_consts, "integrator": m_integrator,
            "delay": m_delay, "noise": m_noise_amp
        }

        # Setup string description
        sys_desc = f"{m_order}"
        if m_integrator: sys_desc += " with Integrator"
        st.markdown(f"### Model Preview: {sys_desc}")
        
        # Calculate Open Loop Step Response
        t_open = np.linspace(0, 200 + m_delay, 1000)
        dt_o = t_open[1] - t_open[0]
        
        # Initialize state vector
        num_states = len(time_consts) + (1 if m_integrator else 0)
        curr_state = np.zeros(num_states)
        
        # Delay buffer
        nd_o = int(max(1, np.ceil(m_delay / dt_o)))
        rb_o = np.zeros(nd_o); idx_o = 0
        pv_o = []
        
        for _ in t_open:
            u_del = rb_o[idx_o]
            rb_o[idx_o] = 1.0 # Unit Step input
            idx_o = (idx_o + 1) % nd_o
            
            step_o = odeint(system_dynamics_generic, curr_state, [0, dt_o], 
                           args=(u_del, m_v, time_consts, m_integrator))
            
            # Add Noise
            noise = np.random.normal(0, m_noise_amp) if m_noise_amp > 0 else 0.0
            pv_o.append(step_o[-1, -1] + noise) # Last state is output
            
            curr_state = step_o[-1]
        
        # Store for CSV
        df_model = pd.DataFrame({"Time": t_open, "Step_Response": pv_o})
        st.session_state['data_model'] = df_model

        fig_open = go.Figure()
        fig_open.add_trace(go.Scatter(x=t_open, y=pv_o, name="Step Response", line=dict(color='#00CC96', width=2)))
        fig_open.update_layout(title="Open Loop Step Response", xaxis_title="Time [s]", yaxis_title="Output y(t)", template="plotly_dark", height=600)
        st.plotly_chart(fig_open, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: TUNING
    # ----------------------------------------------------
    with tab2:
        with st.sidebar:
            st.header("2. Autotuning Setup")
            t_mode_sel = st.selectbox("Algorithm", options=[1, 2, 3], index=2,
                                      format_func=lambda x: {1: "Phase 1 Only", 2: "Symmetric", 3: "Automatic"}[x], key="t_mode_sel")
            if t_mode_sel == 3: t_min_qual = st.slider("Min. Symmetry Quality [%]", 50, 100, 95, key="t_min_qual")
            else: t_min_qual = 95.0
            st.divider()
            t_set = st.number_input("Setpoint", value=150.0, key="t_set")
            t_hyst = st.number_input("Hysteresis", value=1.0, key="t_hyst")
            t_max = st.number_input("Relay Max", value=100.0, key="t_max")
            t_min = st.number_input("Relay Min", value=-100.0, key="t_min")
            t_periods = st.slider("Required Periods", 2, 10, 4, key="t_periods")

        st.subheader("Autotuning Simulation")
        start_tuning = st.button("▶️ Start Tuning", type="primary")

        if start_tuning:
            Ts = 0.05
            t_sim = np.arange(0, 1000, Ts) 
            tuner = OscillationTuning(Ts, t_set, t_min, t_max, t_periods, t_hyst, t_mode_sel, t_min_qual)
            
            y_hist, u_hist = [], []
            
            # Init States
            num_states = len(model_params['time_constants']) + (1 if model_params['integrator'] else 0)
            curr_state = np.zeros(num_states)
            y_prev = 0.0
            
            nd_t = int(max(1, np.ceil(m_delay / Ts)))
            rb_t = np.zeros(nd_t); idx_t = 0
            
            progress = st.progress(0)
            for i, _ in enumerate(t_sim):
                noise = np.random.normal(0, model_params['noise']) if model_params['noise'] > 0 else 0.0
                y_measured = y_prev + noise
                
                u_cmd = tuner.step(y_measured)
                
                u_delayed = rb_t[idx_t]
                rb_t[idx_t] = u_cmd
                idx_t = (idx_t + 1) % nd_t

                y_step = odeint(system_dynamics_generic, curr_state, [0, Ts], 
                                args=(u_delayed, m_v, model_params['time_constants'], model_params['integrator']))
                
                y_hist.append(y_measured); u_hist.append(u_cmd)
                y_prev = y_step[-1, -1]; curr_state = y_step[-1]
                
                if i % 50 == 0: progress.progress(min(i / len(t_sim), 1.0))
                if tuner.tuning_done: break
            progress.progress(100)
            
            # Store & Plot
            t_axis = np.arange(len(y_hist)) * Ts
            st.session_state['data_tuning'] = pd.DataFrame({"Time": t_axis, "PV": y_hist, "SP": [t_set]*len(y_hist), "OP": u_hist})
            st.session_state.kp, st.session_state.tn, st.session_state.tv = tuner.kp, tuner.tn, tuner.tv
            
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

            fig_t = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_t.add_trace(go.Scatter(x=t_axis, y=y_hist, name="PV", line=dict(color='cyan')), row=1, col=1)
            fig_t.add_trace(go.Scatter(x=t_axis, y=[t_set]*len(t_axis), name="SP", line=dict(dash='dash', color='red')), row=1, col=1)
            fig_t.add_trace(go.Scatter(x=t_axis, y=u_hist, name="OP", line=dict(shape='hv', color='orange')), row=2, col=1)
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
            
            rk_type = st.selectbox("Type", ["PID", "PI", "P", "PD"], index=0, key="rk_type")
            
            def_vals = (st.session_state.kp, st.session_state.tn, st.session_state.tv) if use_tuned else (1.0, 5.0, 0.0)
            rk_p = st.number_input("Kp", value=def_vals[0], format="%.4f", key="rk_p")
            rk_n = st.number_input("Tn", value=def_vals[1], format="%.4f", key="rk_n")
            rk_v = st.number_input("Tv", value=def_vals[2], format="%.4f", key="rk_v")
            rk_tf = rk_v / 10.0 if rk_v > 0 else 0.0
            
            st.markdown("---")
            m_sim_time = st.slider("Duration [s]", 10, 500, 200, key="m_sim_time")
            step_val = st.number_input("Step Value", value=st.session_state.get('t_set', 150.0), key="step_val")

        st.subheader("Closed-Loop Control Performance")
        if st.button("🚀 Simulate Step Response", type="primary"):
            t_p = np.linspace(0, m_sim_time, int(m_sim_time * 20))
            sp_vector = np.ones(len(t_p)) * step_val
            
            pid_conf = PIDConfig(rk_p, rk_n, rk_v, rk_tf, rk_type, "ON", "OFF", sp_vector)
            pv, op, _, _, _, _ = simulate_pid_response(t_p, model_params, pid_conf)
            
            st.session_state['data_pid'] = pd.DataFrame({"Time": t_p, "PV": pv, "SP": sp_vector, "OP": op})
            
            # Analysis
            max_pv = np.max(pv)
            overshoot = (max_pv - step_val) / step_val * 100 if max_pv > step_val else 0.0
            
            res1, res2, res3 = st.columns(3)
            res1.metric("Max. Overshoot", f"{overshoot:.2f}%", delta_color="inverse")
            res2.metric("Steady-State Value", f"{pv[-1]:.2f}")
            res3.metric("Control Deviation", f"{step_val - pv[-1]:.4f}")

            fig_pid = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
            fig_pid.add_trace(go.Scatter(x=t_p, y=pv, name="PV", line=dict(color='#00CC96', width=2)), row=1, col=1)
            fig_pid.add_trace(go.Scatter(x=t_p, y=sp_vector, name="SP", line=dict(dash='dash', color='red')), row=1, col=1)
            fig_pid.add_trace(go.Scatter(x=t_p, y=op, name="OP", fill='tozeroy', line=dict(color='gray')), row=2, col=1)
            fig_pid.update_layout(height=700, template="plotly_dark")
            st.plotly_chart(fig_pid, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: HELP & THEORY
    # ----------------------------------------------------
    with tab4:
        st.markdown("""
        # 📚 Documentation & Help

        ## 1. 🎓 Theoretical Background: Relay Auto-Tuning
        
        
        **What is Relay Auto-Tuning?**
        Relay Auto-Tuning, introduced by **Åström and Hägglund (1984)**, is a robust method to identify process characteristics without requiring a complex mathematical model. Instead of a PID controller, a **non-linear relay** (Bang-Bang controller) is inserted into the control loop.
        
        **How does it work?**
        1.  The relay switches the control output between a maximum and minimum value (e.g., +100% / -100%) based on the process variable crossing the setpoint.
        2.  This forces the system into a **Limit Cycle Oscillation**.
        
        3.  The frequency and amplitude of this oscillation reveal the "Ultimate Gain" ($K_u$) and "Ultimate Period" ($T_u$).
        
        **Key Formulas (Describing Function Analysis):**
        The critical point on the Nyquist curve is estimated using:
        
        $$ K_u = \\frac{4d}{\\pi a} $$
        
        Where:
        * $d$: Amplitude of the relay output step (e.g., 100).
        * $a$: Amplitude of the process variable oscillation (PV).
        * $T_u$: The time period of one full oscillation cycle.
        
        **PID Calculation (Ziegler-Nichols Heuristics):**
        
        Once $K_u$ and $T_u$ are found, standard tuning rules are applied:
        * $K_p = 0.6 \cdot K_u$
        * $T_n = 0.5 \cdot T_u$
        * $T_v = 0.125 \cdot T_u$
        
        ---

        ## 2. 📖 User Manual: How to use this App
        
        ### **Step 1: Configure the Plant Model (Tab 1)**
        Define the physical system you want to control.
        * **System Structure:** Choose between 1st, 2nd, or 3rd order systems.
        * **Integrator:** Check this if your system has integrating behavior (e.g., filling a tank without an outlet).
        * **Gain (K):** How sensitive the system is. High gain = large reaction to small input.
        * **Time Constants (T1, T2, T3):** How fast the system reacts. Small T = fast system.
        * **Dead Time:** The delay before the system starts reacting.
        * **Noise:** Simulates real-world sensor noise.
        
        ### **Step 2: Run the Tuner (Tab 2)**
        Start the identification process.
        * **Algorithm:**
            * *Phase 1 Only:* Standard relay test. Fast, but less accurate for asymmetric processes.
            * *Symmetric:* Adds a second phase to find the static load bias. More accurate.
            * *Automatic:* Smartly decides if Phase 2 is needed based on signal symmetry.
        * **Relay Parameters:** Define the limits (Max/Min) and the Setpoint.
        * **Click `▶️ Start Tuning`:** Watch the graph. If successful, suggested PID parameters ($K_p, T_n, T_v$) will appear.
        
        ### **Step 3: Test the Controller (Tab 3)**
        Verify the tuning results in a closed-loop simulation.
        * **Use Tuned Parameters:** Automatically loads the values found in Tab 2. Uncheck to edit them manually.
        * **Controller Type:** Switch between P, PI, PD, or PID.
        * **Click `🚀 Simulate`:** Observe the Step Response.
        * **Metrics:**
            * *Overshoot:* How much the PV exceeds the setpoint.
            * *Steady-State Value:* Where the PV settles.
            * *Deviation:* The remaining error (should be 0 for controllers with I-component).
        
        ### **💾 Saving & Loading**
        Use the sidebar to **Download** your current configuration (JSON) or **Export** all simulation data (CSV) for further analysis in Excel or Matlab.
        """)

    # ----------------------------------------------------
    # GLOBAL EXPORT & SAVE/LOAD
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Data (CSV)")
    
    if any(k in st.session_state for k in ['data_model', 'data_tuning', 'data_pid']):
        frames = []
        if 'data_model' in st.session_state: 
            df = st.session_state['data_model'].copy().add_prefix("Model_"); frames.append(df)
        if 'data_tuning' in st.session_state: 
            df = st.session_state['data_tuning'].copy().add_prefix("Tuning_"); frames.append(df)
        if 'data_pid' in st.session_state: 
            df = st.session_state['data_pid'].copy().add_prefix("PID_"); frames.append(df)
            
        csv_data = pd.concat(frames, axis=1).to_csv(index=False).encode('utf-8')
        st.download_button("Download Master CSV", csv_data, "sim_results.csv", "text/csv")
    else:
        st.info("Run simulations to enable export.")

    with st.sidebar:
        st.markdown("---")
        st.header("💾 File Operations")
        keys = ["m_order", "m_integrator", "m_v", "m_t1", "m_t2", "m_t3", "m_delay", "m_noise_amp", 
                "t_mode_sel", "t_min_qual", "t_set", "t_hyst", "t_max", "t_min", "t_periods", 
                "use_tuned", "rk_type", "rk_p", "rk_n", "rk_v", "m_sim_time", "step_val",
                "kp", "tn", "tv"]
        
        curr_conf = {k: st.session_state[k] for k in keys if k in st.session_state}
        st.download_button("📥 Save Config", json.dumps(curr_conf, indent=2), "config.json", "application/json")
        st.file_uploader("📤 Load Config", type=["json"], key="uploader_key", on_change=load_config_callback)

if __name__ == "__main__":
    main()