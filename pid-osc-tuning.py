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
# --- 0. PAGE CONFIG & SESSION STATE SETUP ---
# ==========================================
st.set_page_config(page_title="PID Control - Oscillation Tuning", layout="wide", page_icon="🎛️")

# Helper: Session State Init
def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# Initialisiere alle wichtigen Keys sofort
init_state('kp', 2.0)
init_state('tn', 10.0)
init_state('tv', 1.0)
init_state('rk_p', 2.0)
init_state('rk_n', 10.0)
init_state('rk_v', 1.0)
init_state('use_tuned', True)

# ==========================================
# --- 1. LOGIC & CALLBACKS ---
# ==========================================

def sync_tuning_params():
    """
    Callback: Synchronisiert die berechneten Tuning-Werte (kp, tn, tv)
    mit den Eingabefeldern der Simulation (rk_p, rk_n, rk_v).
    """
    if st.session_state.use_tuned:
        st.session_state.rk_p = st.session_state.kp
        st.session_state.rk_n = st.session_state.tn
        st.session_state.rk_v = st.session_state.tv

def load_config_callback():
    """Lädt Konfiguration aus hochgeladener Datei."""
    if st.session_state.uploader_key is not None:
        try:
            uploaded_file = st.session_state.uploader_key
            data = json.load(uploaded_file)
            for k, v in data.items():
                st.session_state[k] = v
            sync_tuning_params()
            st.toast("✅ Konfiguration erfolgreich geladen!", icon="💾")
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")

# ==========================================
# --- 2. CLASS DEFINITIONS ---
# ==========================================

class TuningState(IntEnum):
    READY = 1        
    OSCILLATION = 2  

class TuningMode(IntEnum):
    PHASE_1_ONLY = 1    
    ALWAYS_PHASE_2 = 2  
    AUTO = 3            

class OscillationTuning:
    """
    Implements a Relay Auto-Tuning method (Åström & Hägglund).
    """
    def __init__(self, sampling_time: float, setpoint: float, min_out: float, max_out: float, 
                 num_periods: int = 3, min_osc_amp: float = 1.0, tuning_mode: int = 3, min_quality: float = 95.0):
        self.dt = sampling_time
        self.setpoint = setpoint
        self.min_out_limit = min_out
        self.max_out_limit = max_out
        
        self.state = TuningState.READY
        self.tuning_time = 0.0
        self.min_osc_amp = min_osc_amp 
        self.target_periods = num_periods 
        self.tuning_mode = TuningMode(tuning_mode)
        self.min_quality = min_quality    
        
        self.current_phase = 1 
        self.relay_high = max_out
        self.relay_low = min_out
        
        self.quality_area = 0.0 
        self.p1_area = None  # Stores quality of Phase 1 specifically
        self.kp, self.tn, self.tv = 0.0, 0.0, 0.0 
        self.tuning_done = False
        self._reset_oscillation_data()

    def _reset_oscillation_data(self):
        self.num_oscillations = 0
        self.first_setpoint_crossing = False 
        self.max_vals = np.full(self.target_periods + 2, -np.inf)
        self.min_vals = np.full(self.target_periods + 2, np.inf)
        self.max_times = np.zeros(self.target_periods + 2)
        
        self.out_sum = 0.0
        self.out_count = 0
        self.area_pos = 0.0
        self.area_neg = 0.0
        self.output = 0.0

    def check_quality(self):
        max_area = max(self.area_pos, self.area_neg)
        if max_area > 1e-6:
            self.quality_area = (min(self.area_pos, self.area_neg) / max_area) * 100.0
        else:
            self.quality_area = 0.0
        
        finish_tuning = False
        start_phase_2 = False

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
            # If we finish in Phase 1, store the quality as p1_area so UI can show it
            if self.current_phase == 1:
                self.p1_area = self.quality_area
            self._calculate_parameters()
        elif start_phase_2:
            # Store Phase 1 quality before resetting for Phase 2
            self.p1_area = self.quality_area
            self.current_phase = 2
            
            # Calculate bias
            avg_out = self.out_sum / (self.out_count if self.out_count > 0 else 1)
            delta_out = min(self.max_out_limit - avg_out, avg_out - self.min_out_limit)
            self.relay_high = avg_out + delta_out
            self.relay_low = avg_out - delta_out
            self._reset_oscillation_data()
            self.state = TuningState.OSCILLATION

    def _calculate_parameters(self):
        idx = self.num_oscillations
        tu = (self.max_times[idx] - self.max_times[idx-1]) 
        
        if idx > 0:
            avg_max = np.mean(self.max_vals[1:idx+1])
            avg_min = np.mean(self.min_vals[1:idx+1])
            avg_amp = (avg_max - avg_min) / 2.0 
        else:
            avg_amp = 1.0
            
        h = (self.relay_high - self.relay_low) / 2.0
        ku = (4.0 * h) / (np.pi * avg_amp) if avg_amp > 0 else 0
        
        self.kp, self.tn, self.tv = 0.6 * ku, 0.5 * tu, 0.125 * tu
        self.tuning_done = True
        self.state = TuningState.READY

    def step(self, actual_value: float) -> float:
        self.tuning_time += self.dt
        
        if self.state == TuningState.READY:
            self._reset_oscillation_data()
            self.state = TuningState.OSCILLATION
            self.output = self.relay_high if actual_value <= self.setpoint else self.relay_low

        elif self.state == TuningState.OSCILLATION:
            if not self.first_setpoint_crossing:
                if (self.output == self.relay_high and actual_value >= self.setpoint) or \
                   (self.output == self.relay_low and actual_value <= self.setpoint):
                    self.first_setpoint_crossing = True
            
            idx = min(self.num_oscillations, len(self.max_vals)-1)
            if actual_value > self.max_vals[idx]:
                self.max_vals[idx] = actual_value
                self.max_times[idx] = self.tuning_time
            if self.first_setpoint_crossing:
                if actual_value < self.min_vals[idx]:
                    self.min_vals[idx] = actual_value

            if self.num_oscillations >= 1:
                error = actual_value - self.setpoint
                if error > 0: self.area_pos += error * self.dt
                else: self.area_neg += abs(error) * self.dt

            control_deviation = self.setpoint - actual_value
            if control_deviation > self.min_osc_amp: 
                self.output = self.relay_high
            elif control_deviation < -self.min_osc_amp: 
                self.output = self.relay_low

            if self.num_oscillations >= 1:
                self.out_sum += self.output
                self.out_count += 1

            if (self.max_vals[idx] > (self.setpoint + self.min_osc_amp) and 
                self.min_vals[idx] < (self.setpoint - self.min_osc_amp)):
                if self.output == self.relay_high and control_deviation > 0: 
                     if self.num_oscillations < self.target_periods: 
                         self.num_oscillations += 1
                     else:
                         self.check_quality()
                         self.output = 0.0
        return self.output

# ==========================================
# --- 3. STREAMLIT HELPERS & LOGIC ---
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
    derivatives = []
    current_input = u
    state_idx = 0
    
    if has_integrator:
        derivatives.append(current_input)
        current_input = x[state_idx] 
        state_idx += 1
    else:
        current_input = u * gain

    for i, T in enumerate(time_constants):
        y = x[state_idx]
        input_val = current_input * gain if (i == 0 and has_integrator) else current_input
        dydt = (input_val - y) / T
        derivatives.append(dydt)
        current_input = y 
        state_idx += 1
        
    return derivatives

@st.cache_data(show_spinner=False)
def simulate_pid_response(time_vector, model_params, _pid_conf):
    """Simulates the closed-loop response. Cached for performance."""
    ns = len(time_vector) - 1
    dt = time_vector[1] - time_vector[0]
    
    tv_eff = _pid_conf.tv * (_pid_conf.tf / dt * (1.0 - np.exp(-dt / _pid_conf.tf))) if _pid_conf.tf > 0 else _pid_conf.tv
    
    op, e, ie = np.zeros(ns + 1), np.zeros(ns + 1), np.zeros(ns + 1)
    p_term, i_term, d_term = np.zeros(ns + 1), np.zeros(ns + 1), np.zeros(ns + 1)
    
    time_constants = model_params['time_constants']
    has_integrator = model_params['integrator']
    num_states = len(time_constants) + (1 if has_integrator else 0)
    
    pv_output = np.zeros(ns + 1)
    curr_state = np.zeros(num_states)
    
    delay_samples = int(max(1, np.ceil(model_params['delay'] / dt))) 
    ring_buffer = np.zeros(delay_samples)
    buffer_idx = 0

    for i in range(ns):
        pv_val = curr_state[-1] if num_states > 0 else 0.0
        noise_amp = model_params['noise']
        noise = np.random.normal(0, noise_amp) if noise_amp > 0 else 0.0
        pv_measured = pv_val + noise
        pv_output[i] = pv_measured 

        e[i] = _pid_conf.setpoints[i] - pv_measured
        
        if _pid_conf.ctrl_type in ["P", "PI", "PD", "PID"]: 
            p_term[i] = _pid_conf.kp * e[i]
        
        if _pid_conf.ctrl_type in ["PI", "PID"]:
            prev_ie = ie[i-1] if i >= 1 else 0.0
            ie[i] = prev_ie + e[i] * dt
            i_term[i] = (_pid_conf.kp / _pid_conf.tn) * ie[i] if _pid_conf.tn > 0 else 0
        
        if _pid_conf.ctrl_type in ["PD", "PID"] and i >= 1:
            delta_input = (e[i] - e[i-1]) if _pid_conf.d_kickoff == "ON" else (-pv_measured + (pv_output[i-1] if i>0 else 0))
            decay = np.exp(-dt / _pid_conf.tf)
            d_term[i] = (_pid_conf.kp * tv_eff / _pid_conf.tf) * delta_input + d_term[i-1] * decay

        raw_op = p_term[i] + i_term[i] + d_term[i]
        op[i] = np.clip(raw_op, -100.0, 100.0) 
        
        if _pid_conf.anti_windup == "ON":
            is_sat = (raw_op > 100.0 and e[i] > 0) or (raw_op < -100.0 and e[i] < 0)
            if is_sat:
                ie[i] -= e[i] * dt 
                i_term[i] = (_pid_conf.kp / _pid_conf.tn) * ie[i] if _pid_conf.tn > 0 else 0
        
        delayed_u = ring_buffer[buffer_idx]
        ring_buffer[buffer_idx] = op[i]
        buffer_idx = (buffer_idx + 1) % delay_samples
        
        step_res = odeint(system_dynamics_generic, curr_state, [0, dt], 
                        args=(delayed_u, model_params['gain'], time_constants, has_integrator))
        
        curr_state = step_res[-1]
        
    pv_output[-1] = curr_state[-1] if num_states > 0 else 0.0
        
    return pv_output, op, e, p_term, i_term, d_term

# ==========================================
# --- 4. MAIN UI STRUCTURE ---
# ==========================================

def main():
    st.markdown("""
    <style>
        button[data-baseweb="tab"] { font-size: 20px !important; font-weight: bold !important; }
        .stMetric { background-color: #0E1117; padding: 10px; border-radius: 5px; border: 1px solid #303030; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🕹️ Control Engineering Dashboard")

    # TABS DEFINITION
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
            
            m_order = st.selectbox("System Structure", ["1st Order System", "2nd Order System", "3rd Order System"], index=1, key="m_order")
            m_v = st.slider("Gain (K)", 0.1, 10.0, 2.0, 0.1, key="m_v")
            m_integrator = False
            
            time_consts = []
            m_t1 = st.slider("Time Constant 1 (T1)", 0.1, 20.0, 10.0, 0.1, key="m_t1")
            time_consts.append(m_t1)
            
            if m_order == "1st Order System":
                m_integrator = st.checkbox("Set Integrator", value=False, key="m_integrator")
            
            if m_order in ["2nd Order System", "3rd Order System"]:
                m_t2 = st.slider("Time Constant 2 (T2)", 0.1, 20.0, 1.0, 0.1, key="m_t2")
                time_consts.append(m_t2)
                
            if m_order == "3rd Order System":
                m_t3 = st.slider("Time Constant 3 (T3)", 0.1, 20.0, 0.5, 0.1, key="m_t3")
                time_consts.append(m_t3)
            
            m_delay = st.slider("Dead Time [s]", 0.0, 10.0, 0.0, key="m_delay")
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
        
        # Calculate Open Loop Step Response (Simplified in-line simulation)
        t_open = np.linspace(0, 200 + m_delay, 500)
        dt_o = t_open[1] - t_open[0]
        num_states = len(time_consts) + (1 if m_integrator else 0)
        curr_state = np.zeros(num_states)
        nd_o = int(max(1, np.ceil(m_delay / dt_o)))
        rb_o = np.zeros(nd_o); idx_o = 0
        pv_o = []
        
        for _ in t_open:
            u_del = rb_o[idx_o]
            rb_o[idx_o] = 1.0 
            idx_o = (idx_o + 1) % nd_o
            step_o = odeint(system_dynamics_generic, curr_state, [0, dt_o], 
                           args=(u_del, m_v, time_consts, m_integrator))
            noise = np.random.normal(0, m_noise_amp) if m_noise_amp > 0 else 0.0
            pv_o.append(step_o[-1, -1] + noise)
            curr_state = step_o[-1]
        
        df_model = pd.DataFrame({"Time": t_open, "Step_Response": pv_o})
        st.session_state['data_model'] = df_model

        fig_open = go.Figure()
        fig_open.add_trace(go.Scatter(x=t_open, y=pv_o, name="Step Response", line=dict(color='#00CC96', width=2)))
        fig_open.update_layout(title="Open Loop Step Response (Unit Step)", xaxis_title="Time [s]", yaxis_title="Output y(t)", template="plotly_dark", height=500)
        st.plotly_chart(fig_open, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: TUNING
    # ----------------------------------------------------
    with tab2:
        with st.sidebar:
            st.header("2. Autotuning Setup")
            # --- NAMING CHANGED AS REQUESTED ---
            t_mode_sel = st.selectbox("Algorithm", options=[1, 2, 3], index=2,
                                    format_func=lambda x: {1: "Phase 1", 2: "Phase 1+2", 3: "Automatic"}[x], key="t_mode_sel")
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
                
                if i % 100 == 0: progress.progress(min(i / len(t_sim), 1.0))
                if tuner.tuning_done: break
            progress.progress(100)
            
            # Store & Plot
            t_axis = np.arange(len(y_hist)) * Ts
            st.session_state['data_tuning'] = pd.DataFrame({"Time": t_axis, "PV": y_hist, "SP": [t_set]*len(y_hist), "OP": u_hist})
            
            st.session_state.kp = tuner.kp
            st.session_state.tn = tuner.tn
            st.session_state.tv = tuner.tv
            
            # Sync to controller settings
            sync_tuning_params()
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Recommended Kp", f"{tuner.kp:.3f}")
            c2.metric("Recommended Tn", f"{tuner.tn:.3f}s")
            c3.metric("Recommended Tv", f"{tuner.tv:.3f}s")
            
            # --- PHASE 1 QUALITY DISPLAY ---
            if tuner.p1_area is not None:
                p1_val = tuner.p1_area
                p1_delta = "Good" if p1_val > 90 else "Low"
                c4.metric("Phase 1 Quality", f"{p1_val:.1f}%", delta=p1_delta)
            else:
                c4.metric("Phase 1 Quality", "N/A")

            # --- PHASE 2 QUALITY DISPLAY ---
            if tuner.current_phase == 2:
                final_qual = tuner.quality_area
                final_delta = "Good" if final_qual > 90 else "Low"
                c5.metric("Phase 2 Quality", f"{final_qual:.1f}%", delta=final_delta)
            else:
                 c5.metric("Phase 2 Quality", "Not active", delta_color="off")

            fig_t = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_t.add_trace(go.Scatter(x=t_axis, y=y_hist, name="PV", line=dict(color='cyan')), row=1, col=1)
            fig_t.add_trace(go.Scatter(x=t_axis, y=[t_set]*len(t_axis), name="SP", line=dict(dash='dash', color='red')), row=1, col=1)
            fig_t.add_trace(go.Scatter(x=t_axis, y=u_hist, name="OP", line=dict(shape='hv', color='orange')), row=2, col=1)
            fig_t.update_layout(height=500, template="plotly_dark", title="Tuning Progress")
            st.plotly_chart(fig_t, use_container_width=True)
            
            st.success("Tuning Complete! Parameters updated in Controller Test Tab.")

    # ----------------------------------------------------
    # TAB 3: PID TEST
    # ----------------------------------------------------
    with tab3:
        with st.sidebar:
            st.header("3. Controller Test")
            
            # Checkbox with Callback
            use_tuned = st.checkbox("Use Tuned Parameters", key="use_tuned", on_change=sync_tuning_params)
            
            st.markdown("---")
            rk_type = st.selectbox("Type", ["PID", "PI", "P", "PD"], index=0, key="rk_type")
            
            # Use keys directly
            rk_p = st.number_input("Kp", format="%.4f", key="rk_p")
            rk_n = st.number_input("Tn", format="%.4f", key="rk_n")
            rk_v = st.number_input("Tv", format="%.4f", key="rk_v")
            
            rk_tf = rk_v / 10.0 if rk_v > 0 else 0.0
            
            st.markdown("---")
            m_sim_time = st.slider("Duration [s]", 10, 500, 200, key="m_sim_time")
            step_val = st.number_input("Step Value", value=st.session_state.get('t_set', 150.0), key="step_val")

        st.subheader("Closed-Loop Control Performance")
        
        if st.button("🚀 Simulate Step Response", type="primary"):
            t_p = np.linspace(0, m_sim_time, int(m_sim_time * 20))
            sp_vector = np.ones(len(t_p)) * step_val
            
            # Convert pure Python PIDConfig to use cached function
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
            fig_pid.update_layout(height=600, template="plotly_dark")
            st.plotly_chart(fig_pid, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: HELP & THEORY
    # ----------------------------------------------------
    with tab4:
        st.markdown("""
        # 📚 Documentation
        
        
        **Relay Auto-Tuning** (Åström/Hägglund) induces a limit cycle oscillation to identify process characteristics ($K_u$, $T_u$) without a mathematical model.
        
        ### Quick Guide:
        1.  **Tab 1:** Define your Plant (Process).
        2.  **Tab 2:** Click 'Start Tuning'. Wait for PID params.
        3.  **Tab 3:** Parameters are automatically copied if 'Use Tuned Parameters' is checked. Click 'Simulate' to test.
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

    with st.sidebar:
        st.markdown("---")
        st.header("💾 File Operations")
        keys = ["m_order", "m_integrator", "m_v", "m_t1", "m_t2", "m_t3", "m_delay", "m_noise_amp", 
                "t_mode_sel", "t_min_qual", "t_set", "t_hyst", "t_max", "t_min", "t_periods", 
                "use_tuned", "rk_type", "rk_p", "rk_n", "rk_v", "m_sim_time", "step_val"]
        
        curr_conf = {k: st.session_state[k] for k in keys if k in st.session_state}
        st.download_button("📥 Save Config", json.dumps(curr_conf, indent=2), "config.json", "application/json")
        st.file_uploader("📤 Load Config", type=["json"], key="uploader_key", on_change=load_config_callback)

if __name__ == "__main__":
    main()