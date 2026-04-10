import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from enum import IntEnum
import json

# ==========================================
# --- 0. PAGE CONFIG & STATE MANAGEMENT ---
# ==========================================
st.set_page_config(page_title="PID Control - POI Tuning", layout="wide", page_icon="🔬")


def init_state(key, default_value):
    """Initialize session state variable safely."""
    if key not in st.session_state:
        st.session_state[key] = default_value


# --- Plant Model Defaults ---
init_state('m_plant_type', "PT2 with Dead Time")
init_state('m_ks', 2.0)
init_state('m_kis', 0.05)
init_state('m_t1', 20.0)
init_state('m_t2', 10.0)
init_state('m_delay', 5.0)
init_state('m_noise_amp', 0.2)

# --- Tuning Defaults ---
init_state('t_setpoint', 100.0)
init_state('t_y_tune', 100.0)
init_state('t_max', 100.0)
init_state('t_min', 0.0)
init_state('t_tf1', 5.0)
init_state('t_tf2', 3.0)
init_state('t_sim_time', 600)

# --- Identification Results (plant params, not PID) ---
init_state('id_tu', 0.0)
init_state('id_tg', 0.0)
init_state('id_ks', 0.0)
init_state('id_kis', 0.0)
init_state('id_plant_type', "PT2 with Dead Time")

# --- Controller Test ---
init_state('use_tuned', True)
init_state('chr_mode', "20% Overshoot — Setpoint Tracking")
init_state('rk_type', "PID")
init_state('rk_p', 1.0)
init_state('rk_n', 10.0)
init_state('rk_v', 1.0)
init_state('c_sim_time', 600)
init_state('c_step_val', 100.0)


# ==========================================
# --- 1. CALLBACKS ---
# ==========================================

def compute_chr_params(plant_type, Tu, Tg, Ks, Kis, mode):
    """Compute PID parameters using CHR tuning rules.

    Modes:
      - 20% Overshoot, Setpoint Tracking
      - 20% Overshoot, Disturbance Rejection
      - Aperiodic, Setpoint Tracking
      - Aperiodic, Disturbance Rejection

    Returns (Kp, Tn, Tv).
    """
    if plant_type == "PT2 with Dead Time":
        if Tu < 1e-9 or Ks < 1e-9:
            return 0.1, 100.0, 0.0
        if mode == "20% Overshoot — Setpoint Tracking":
            Kp = 0.95 * Tg / (Ks * Tu)
            Tn = 1.35 * Tu
            Tv = 0.47 * Tu
        elif mode == "20% Overshoot — Disturbance Rejection":
            Kp = 1.2 * Tg / (Ks * Tu)
            Tn = 2.3 * Tu
            Tv = 0.42 * Tu
        elif mode == "Aperiodic — Setpoint Tracking":
            Kp = 0.6 * Tg / (Ks * Tu)
            Tn = Tg
            Tv = 0.5 * Tu
        else:  # Aperiodic — Disturbance Rejection
            Kp = 0.95 * Tg / (Ks * Tu)
            Tn = 2.4 * Tu
            Tv = 0.42 * Tu
    else:  # IT1
        if Tu < 1e-9 or Kis < 1e-9:
            return 0.1, 100.0, 0.0
        if mode == "20% Overshoot — Setpoint Tracking":
            Kp = 0.95 / (Kis * Tu)
            Tn = 2.0 * Tu
            Tv = 0.47 * Tu
        elif mode == "20% Overshoot — Disturbance Rejection":
            Kp = 1.2 / (Kis * Tu)
            Tn = 2.0 * Tu
            Tv = 0.42 * Tu
        elif mode == "Aperiodic — Setpoint Tracking":
            Kp = 0.59 / (Kis * Tu)
            Tn = 2.4 * Tu
            Tv = 0.5 * Tu
        else:  # Aperiodic — Disturbance Rejection
            Kp = 0.95 / (Kis * Tu)
            Tn = 2.4 * Tu
            Tv = 0.42 * Tu
    return Kp, Tn, Tv


def sync_tuning_params():
    """Compute CHR PID params from identification results and copy to PID test fields."""
    if st.session_state.use_tuned:
        Kp, Tn, Tv = compute_chr_params(
            st.session_state.id_plant_type,
            st.session_state.id_tu,
            st.session_state.id_tg,
            st.session_state.id_ks,
            st.session_state.id_kis,
            st.session_state.chr_mode)
        st.session_state.rk_p = Kp
        st.session_state.rk_n = Tn
        st.session_state.rk_v = Tv
        st.session_state.c_step_val = st.session_state.t_setpoint


def on_upload_callback():
    """Executed when a config file is uploaded."""
    if st.session_state.uploader_key is not None:
        try:
            data = json.load(st.session_state.uploader_key)
            for k, v in data.items():
                st.session_state[k] = v
            sync_tuning_params()
            st.toast("Configuration loaded!", icon="💾")
        except Exception as e:
            st.error(f"Error loading config: {e}")


# ==========================================
# --- 2. PLANT DYNAMICS ---
# ==========================================

def pt2_dynamics(state, t, u, Ks, T1, T2):
    """PT2: Ks / ((1+s*T1)*(1+s*T2))"""
    x1, x2 = state
    dx1 = (Ks * u - x1) / T1
    dx2 = (x1 - x2) / T2
    return [dx1, dx2]


def it1_dynamics(state, t, u, Kis, T1):
    """IT1: Kis / (s * (1+s*T1))"""
    x1, x2 = state
    dx1 = Kis * u
    dx2 = (x1 - x2) / T1
    return [dx1, dx2]


# ==========================================
# --- 3. POI TUNING CLASS ---
# ==========================================

class TuningState(IntEnum):
    INIT = 0
    HEATING = 1
    IDENTIFIED = 2
    DONE = 3


class POITuner:
    """Point-of-inflection autotuning with filtered derivatives and delay compensation."""

    def __init__(self, dt, Tf1, Tf2, y_tune, setpoint, plant_type):
        self.dt = dt
        self.Tf1 = Tf1
        self.Tf2 = Tf2
        self.y_tune = y_tune
        self.setpoint = setpoint
        self.plant_type = plant_type

        self.state = TuningState.INIT
        self.t = 0.0
        self.x_0 = 0.0

        # Filter states
        self.v_k = 0.0
        self.v_prev = 0.0
        self.a_k = 0.0
        self.x_prev = 0.0

        # Inflection detection (sliding window on a_k / a_max ratio)
        self.v_max_measured = 0.0
        self.t_v_max = 0.0
        self.a_max_measured = 0.0
        self.window_size = max(5, int(2.0 * Tf2 / dt))
        self.window_threshold = 0.8
        self.a_ratio_window = []

        # Ring buffer for delay compensation lookup
        buf_seconds = Tf1 + Tf2 + 120.0
        self.buffer_size = max(10, int(buf_seconds / dt))
        self.x_buffer = np.zeros(self.buffer_size)
        self.t_buffer = np.zeros(self.buffer_size)
        self.buf_idx = 0
        self.buf_count = 0

        # Identification results (plant parameters only, no PID)
        self.Tu = 0.0
        self.t_switch = 0.0
        self.tuning_done = False

        self.t_wp_real = 0.0
        self.x_wp_real = 0.0
        self.v_max_real = 0.0
        self.Ks_est = 0.0
        self.Kis_est = 0.0
        self.Tg_est = 0.0
        self.x_max_est = 0.0

        # History for plots
        self.v_history = []
        self.a_history = []

    def _update_filters(self, x_measured):
        """Exact discrete-time filter equations per specification."""
        Ta = self.dt
        Tf1, Tf2 = self.Tf1, self.Tf2

        self.v_prev = self.v_k

        # Filtered 1st derivative: v_k = Tf1/(Tf1+Ta)*v_{k-1} + 1/(Tf1+Ta)*(x_k - x_{k-1})
        self.v_k = (Tf1 / (Tf1 + Ta)) * self.v_k + (1.0 / (Tf1 + Ta)) * (x_measured - self.x_prev)

        # Filtered 2nd derivative: a_k = Tf2/(Tf2+Ta)*a_{k-1} + 1/(Tf2+Ta)*(v_k - v_{k-1})
        self.a_k = (Tf2 / (Tf2 + Ta)) * self.a_k + (1.0 / (Tf2 + Ta)) * (self.v_k - self.v_prev)

        self.x_prev = x_measured

    def _store_buffer(self, x_measured):
        """Store value in ring buffer."""
        self.x_buffer[self.buf_idx] = x_measured
        self.t_buffer[self.buf_idx] = self.t
        self.buf_idx = (self.buf_idx + 1) % self.buffer_size
        self.buf_count = min(self.buf_count + 1, self.buffer_size)

    def _lookup_buffer(self, t_target):
        """Look up temperature at time t_target from ring buffer."""
        best_idx = 0
        best_diff = float('inf')
        for i in range(self.buf_count):
            idx = (self.buf_idx - self.buf_count + i) % self.buffer_size
            diff = abs(self.t_buffer[idx] - t_target)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return self.x_buffer[best_idx]

    def _compute_parameters(self):
        """Compensate filter delay and compute PID parameters."""
        self.state = TuningState.IDENTIFIED
        self.tuning_done = True
        self.t_switch = self.t

        # True inflection point time (compensate filter delay)
        self.t_wp_real = self.t_v_max - (self.Tf1 + self.Tf2)
        self.t_wp_real = max(0.0, self.t_wp_real)

        # True temperature at inflection point from ring buffer
        self.x_wp_real = self._lookup_buffer(self.t_wp_real)
        self.v_max_real = self.v_max_measured

        # Dead time Tu (geometric tangent construction)
        if self.v_max_real > 1e-9:
            self.Tu = self.t_wp_real - (self.x_wp_real - self.x_0) / self.v_max_real
        else:
            self.Tu = 1.0
        self.Tu = max(0.1, self.Tu)

        if self.plant_type == "PT2 with Dead Time":
            # Estimate steady-state (inflection at ~28% of total rise)
            self.x_max_est = self.x_0 + (self.x_wp_real - self.x_0) / 0.28
            self.Ks_est = (self.x_max_est - self.x_0) / self.y_tune
            self.Tg_est = (self.x_max_est - self.x_0) / self.v_max_real
        else:
            # IT1: integration coefficient
            self.Kis_est = self.v_max_real / self.y_tune

    def step(self, x_measured):
        """One step of the state machine. Returns the control output."""
        self.t += self.dt

        if self.state == TuningState.INIT:
            self._update_filters(x_measured)
            self.v_history.append(self.v_k)
            self.a_history.append(self.a_k)

            # Wait until process is quiet
            if self.t >= 1.0:
                self.x_0 = x_measured
                self.x_prev = x_measured
                self.v_k = 0.0
                self.a_k = 0.0
                self.v_prev = 0.0
                self.state = TuningState.HEATING
            return 0.0

        elif self.state == TuningState.HEATING:
            self._store_buffer(x_measured)
            self._update_filters(x_measured)
            self.v_history.append(self.v_k)
            self.a_history.append(self.a_k)

            # Track maximum slope and acceleration
            if self.v_k > self.v_max_measured:
                self.v_max_measured = self.v_k
                self.t_v_max = self.t
            if self.a_k > self.a_max_measured:
                self.a_max_measured = self.a_k

            # Start searching after filters have settled
            min_wait = self.Tf1 + self.Tf2
            if self.t > min_wait and self.a_max_measured > 1e-9:
                # Criterion: a_k has dropped below 10% of its peak
                # Works for PT2 (a goes negative) and IT1 (a falls to ~0)
                hit = 1 if self.a_k < 0.1 * self.a_max_measured else 0
                self.a_ratio_window.append(hit)
                if len(self.a_ratio_window) > self.window_size:
                    self.a_ratio_window.pop(0)

                # Confirmed when 80% of the window meets the criterion
                # and v_k is significant (not a noise artifact)
                if len(self.a_ratio_window) >= self.window_size:
                    ratio = sum(self.a_ratio_window) / len(self.a_ratio_window)
                    if ratio >= self.window_threshold and self.v_k > 0.05 * self.v_max_measured:
                        self._compute_parameters()
                        return self.y_tune

            return self.y_tune

        elif self.state == TuningState.IDENTIFIED:
            # Keep heating: PT2 until steady-state, IT1 until setpoint
            self._update_filters(x_measured)
            self.v_history.append(self.v_k)
            self.a_history.append(self.a_k)

            if self.plant_type == "PT2 with Dead Time":
                if abs(self.v_k) < 0.01 * self.v_max_measured and self.t > self.t_switch + 10.0:
                    self.state = TuningState.DONE
            else:
                if x_measured >= self.setpoint:
                    self.state = TuningState.DONE
            return self.y_tune

        else:
            # DONE: output stays at y_tune (step input)
            self._update_filters(x_measured)
            self.v_history.append(self.v_k)
            self.a_history.append(self.a_k)
            return self.y_tune


# ==========================================
# --- 4. SIMULATION FUNCTIONS ---
# ==========================================

def simulate_tuning(model_params, tuning_params):
    """Run pure open-loop tuning simulation (no PID)."""
    dt = 0.1
    t_max = tuning_params['sim_time']
    steps = int(t_max / dt)

    plant_type = model_params['plant_type']
    setpoint = tuning_params['setpoint']
    y_tune = tuning_params['y_tune']

    tuner = POITuner(dt, tuning_params['Tf1'], tuning_params['Tf2'],
                     y_tune, setpoint, plant_type)

    # Dead time buffer
    delay_samples = max(1, int(model_params['delay'] / dt))
    delay_buf = [0.0] * delay_samples

    # Plant state
    curr_state = np.zeros(2)

    # Fixed arrays for full simulation duration
    t_arr = np.zeros(steps)
    x_arr = np.zeros(steps)
    y_arr = np.zeros(steps)

    for i in range(steps):
        t_arr[i] = i * dt

        # Measurement (with noise)
        pv_true = curr_state[-1]
        noise = np.random.normal(0, model_params['noise']) if model_params['noise'] > 0 else 0.0
        x_measured = pv_true + noise
        x_arr[i] = x_measured

        # Tuner computes control output (open-loop)
        y_arr[i] = tuner.step(x_measured)

        # Dead time
        y_delayed = delay_buf.pop(0)
        delay_buf.append(y_arr[i])

        # Integrate plant
        if plant_type == "PT2 with Dead Time":
            res = odeint(pt2_dynamics, curr_state, [0, dt],
                         args=(y_delayed, model_params['Ks'], model_params['T1'], model_params['T2']))
        else:
            res = odeint(it1_dynamics, curr_state, [0, dt],
                         args=(y_delayed, model_params['Kis'], model_params['T1']))
        curr_state = res[-1]

    w_arr = np.full(steps, setpoint)

    v_arr = np.array(tuner.v_history[:steps])
    a_arr = np.array(tuner.a_history[:steps])
    if len(v_arr) < steps:
        v_arr = np.pad(v_arr, (0, steps - len(v_arr)))
        a_arr = np.pad(a_arr, (0, steps - len(a_arr)))

    return t_arr, x_arr, y_arr, w_arr, v_arr, a_arr, tuner


@st.cache_data(show_spinner=False)
def simulate_pid_closedloop(model_params_tuple, pid_kp, pid_tn, pid_tv, pid_type,
                            setpoint, sim_time, out_min, out_max):
    """Standalone closed-loop PID simulation for Tab 3."""
    model_params = json.loads(model_params_tuple)
    dt = 0.1
    steps = int(sim_time / dt)

    Tf_d = pid_tv / 10.0 if pid_tv > 0 else 0.1

    delay_samples = max(1, int(model_params['delay'] / dt))
    delay_buf = [0.0] * delay_samples
    curr_state = np.zeros(2)

    t_arr = np.zeros(steps)
    x_arr = np.zeros(steps)
    y_arr = np.zeros(steps)
    w_arr = np.full(steps, setpoint)

    I_term = 0.0
    d_filt = 0.0
    x_prev = 0.0

    for i in range(steps):
        t_arr[i] = i * dt
        pv_true = curr_state[-1]
        noise = np.random.normal(0, model_params['noise']) if model_params['noise'] > 0 else 0.0
        x_measured = pv_true + noise
        x_arr[i] = x_measured

        e = setpoint - x_measured

        # P
        P = pid_kp * e if pid_type in ("P", "PI", "PD", "PID") else 0.0

        # I
        if pid_type in ("PI", "PID") and pid_tn > 0:
            I_term += (pid_kp / pid_tn) * e * dt
            I_term = np.clip(I_term, out_min, out_max)
        I = I_term if pid_type in ("PI", "PID") else 0.0

        # D (on PV, no derivative kick)
        if pid_type in ("PD", "PID") and i > 0:
            dx = -(x_measured - x_prev) / dt
            alpha = dt / (Tf_d + dt)
            d_filt = (1 - alpha) * d_filt + alpha * dx
            D = pid_kp * pid_tv * d_filt
        else:
            D = 0.0

        raw_y = P + I + D
        y = np.clip(raw_y, out_min, out_max)

        # Anti-Windup
        if pid_type in ("PI", "PID") and pid_tn > 0:
            if (raw_y > out_max and e > 0) or (raw_y < out_min and e < 0):
                I_term -= (pid_kp / pid_tn) * e * dt

        y_arr[i] = y
        x_prev = x_measured

        y_delayed = delay_buf.pop(0)
        delay_buf.append(y)

        if model_params['plant_type'] == "PT2 with Dead Time":
            res = odeint(pt2_dynamics, curr_state, [0, dt],
                         args=(y_delayed, model_params['Ks'], model_params['T1'], model_params['T2']))
        else:
            res = odeint(it1_dynamics, curr_state, [0, dt],
                         args=(y_delayed, model_params['Kis'], model_params['T1']))
        curr_state = res[-1]

    return t_arr, x_arr, y_arr, w_arr


# ==========================================
# --- 5. MAIN UI ---
# ==========================================

def main():
    st.markdown("""
    <style>
        button[data-baseweb="tab"] { font-size: 20px !important; font-weight: bold !important; }
        .stMetric { background-color: #0E1117; padding: 10px; border-radius: 5px; border: 1px solid #303030; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 Point-of-Inflection Autotuning")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 🏗️ Plant Model",
        "2. 📈 POI Tuning",
        "3. 🚀 PID Test",
        "4. 📘 Help & Theory"
    ])

    # -----------------------------------------------
    # TAB 1: PLANT MODEL
    # -----------------------------------------------
    with tab1:
        with st.sidebar:
            st.header("1. Plant Parameters")

            m_plant_type = st.radio("Plant Type", ["PT2 with Dead Time", "IT1 with Dead Time"],
                                    key="m_plant_type")

            if m_plant_type == "PT2 with Dead Time":
                m_ks = st.slider("Gain (Ks)", 0.5, 5.0, step=0.1, key="m_ks")
                m_t1 = st.slider("Time Constant T1 (s)", 1.0, 50.0, step=1.0, key="m_t1")
                m_t2 = st.slider("Time Constant T2 (s)", 1.0, 30.0, step=1.0, key="m_t2")
            else:
                m_kis = st.slider("Integration Coeff. (Kis)", 0.01, 0.5, step=0.01, key="m_kis")
                m_t1 = st.slider("Lag T1 (s)", 1.0, 50.0, step=1.0, key="m_t1")

            m_delay = st.slider("Dead Time Tt (s)", 0.0, 20.0, step=0.5, key="m_delay")
            st.divider()
            m_noise_amp = st.slider("Measurement Noise (Amp)", 0.0, 2.0, step=0.1, key="m_noise_amp")

        # Model parameter dictionary
        model_params = {
            'plant_type': st.session_state.m_plant_type,
            'Ks': st.session_state.m_ks,
            'Kis': st.session_state.m_kis,
            'T1': st.session_state.m_t1,
            'T2': st.session_state.m_t2 if st.session_state.m_plant_type == "PT2 with Dead Time" else 1.0,
            'delay': st.session_state.m_delay,
            'noise': st.session_state.m_noise_amp,
        }

        st.markdown(f"### Plant Preview: {model_params['plant_type']}")

        # Open-loop step response
        dt_o = 0.1
        t_open = np.arange(0, 300, dt_o)
        delay_samp = max(1, int(model_params['delay'] / dt_o))
        rb_o = [0.0] * delay_samp
        curr_st = np.zeros(2)
        pv_open = []

        for _ in t_open:
            y_del = rb_o.pop(0)
            rb_o.append(100.0)
            if model_params['plant_type'] == "PT2 with Dead Time":
                res = odeint(pt2_dynamics, curr_st, [0, dt_o],
                             args=(y_del, model_params['Ks'], model_params['T1'], model_params['T2']))
            else:
                res = odeint(it1_dynamics, curr_st, [0, dt_o],
                             args=(y_del, model_params['Kis'], model_params['T1']))
            noise = np.random.normal(0, model_params['noise']) if model_params['noise'] > 0 else 0.0
            pv_open.append(res[-1, -1] + noise)
            curr_st = res[-1]

        df_model = pd.DataFrame({"Time": t_open, "Step_Response": pv_open})
        st.session_state['data_model'] = df_model

        fig_open = go.Figure()
        fig_open.add_trace(go.Scatter(x=t_open, y=pv_open, name="Step Response",
                                      line=dict(color='#00CC96', width=2)))
        fig_open.update_layout(title="Open-Loop Step Response (100% Step)",
                               xaxis_title="Time [s]", yaxis_title="Output y(t)",
                               template="plotly_dark", height=500)
        st.plotly_chart(fig_open, use_container_width=True)

    # -----------------------------------------------
    # TAB 2: POI TUNING
    # -----------------------------------------------
    with tab2:
        with st.sidebar:
            st.header("2. Tuning Setup")
            t_setpoint = st.number_input("Setpoint (w)", key="t_setpoint")
            t_y_tune = st.number_input("Tuning Output (%)", key="t_y_tune")
            st.divider()
            t_max = st.number_input("Output Max", key="t_max")
            t_min = st.number_input("Output Min", key="t_min")
            st.divider()
            t_tf1 = st.slider("Filter Time Tf1 (s)", 1.0, 20.0, step=0.5, key="t_tf1")
            t_tf2 = st.slider("Filter Time Tf2 (s)", 0.5, 10.0, step=0.5, key="t_tf2")
            t_sim_time = st.slider("Sim. Duration [s]", 100, 1200, step=50, key="t_sim_time")

        st.subheader("Point-of-Inflection Autotuning")

        if st.button("▶️ Start Tuning", type="primary"):
            tuning_params = {
                'setpoint': st.session_state.t_setpoint,
                'y_tune': st.session_state.t_y_tune,
                'Tf1': st.session_state.t_tf1,
                'Tf2': st.session_state.t_tf2,
                'sim_time': st.session_state.t_sim_time,
            }

            with st.spinner("Simulation running..."):
                t_arr, x_arr, y_arr, w_arr, v_arr, a_arr, tuner = simulate_tuning(
                    model_params, tuning_params)

            # Store identification results in session state
            st.session_state.id_tu = tuner.Tu
            st.session_state.id_tg = tuner.Tg_est
            st.session_state.id_ks = tuner.Ks_est
            st.session_state.id_kis = tuner.Kis_est
            st.session_state.id_plant_type = tuner.plant_type
            sync_tuning_params()

            # Save data
            st.session_state['data_tuning'] = pd.DataFrame({
                "Time": t_arr, "PV": x_arr, "SP": w_arr, "OP": y_arr,
                "v_filt": v_arr, "a_filt": a_arr
            })

            # Status
            if tuner.tuning_done:
                st.success("Tuning successful! Inflection point identified. Parameters copied to Tab 3.")
            else:
                st.warning("Inflection point not found. Increase simulation duration or adjust filter times.")

            # Inflection point times for markers across all plots
            t_wp = tuner.t_wp_real if tuner.tuning_done else None       # compensated (true)
            t_meas = tuner.t_v_max if tuner.tuning_done else None       # measured (delayed)

            # Look up v/a values at both time points for marker placement
            if tuner.tuning_done:
                dt_sim = t_arr[1] - t_arr[0] if len(t_arr) > 1 else 0.1
                idx_wp = min(int(t_wp / dt_sim), len(v_arr) - 1)
                idx_meas = min(int(t_meas / dt_sim), len(v_arr) - 1)
                v_at_wp = v_arr[idx_wp]
                v_at_meas = v_arr[idx_meas]       # = v_max (peak)
                a_at_wp = a_arr[idx_wp]
                a_at_meas = a_arr[idx_meas]

            # === Plot 1: Open-Loop Step Response ===
            fig_t1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.6, 0.4],
                                   subplot_titles=["Open-Loop Step Response",
                                                   "Control Output"])
            fig_t1.add_trace(go.Scatter(x=t_arr, y=x_arr,
                                        name="PV", line=dict(color='cyan')), row=1, col=1)
            fig_t1.add_trace(go.Scatter(x=t_arr, y=w_arr,
                                        name="SP", line=dict(dash='dot', color='gray')), row=1, col=1)
            fig_t1.add_trace(go.Scatter(x=t_arr, y=y_arr,
                                        name="OP (%)", line=dict(color='orange')), row=2, col=1)
            if t_wp is not None:
                # Measured (delayed) inflection point
                fig_t1.add_vline(x=t_meas, line_dash="dot", line_color="red",
                                 annotation_text="Measured (delayed)")
                fig_t1.add_trace(go.Scatter(
                    x=[t_meas], y=[x_arr[idx_meas]], mode='markers',
                    marker=dict(size=10, color='red', symbol='circle'),
                    name="Measured (delayed)", showlegend=True), row=1, col=1)
                # Compensated (true) inflection point
                fig_t1.add_vline(x=t_wp, line_dash="dash", line_color="yellow",
                                 annotation_text="Compensated (true)")
                fig_t1.add_trace(go.Scatter(
                    x=[t_wp], y=[tuner.x_wp_real], mode='markers',
                    marker=dict(size=12, color='yellow', symbol='x'),
                    name="Compensated (true)", showlegend=True), row=1, col=1)
            fig_t1.update_layout(height=500, template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_t1, use_container_width=True)

            # === Plot 2: DT1 - Filtered Slope v(t) ===
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=t_arr, y=v_arr,
                                       name="v(t)", line=dict(color='#00CC96', width=2)))
            if t_wp is not None:
                # Measured (delayed) - where v peaks in the filtered signal
                fig_v.add_vline(x=t_meas, line_dash="dot", line_color="red",
                                annotation_text="Measured (delayed)")
                fig_v.add_trace(go.Scatter(
                    x=[t_meas], y=[v_at_meas], mode='markers',
                    marker=dict(size=10, color='red', symbol='circle'),
                    name="Measured (delayed)"))
                # Compensated (true)
                fig_v.add_vline(x=t_wp, line_dash="dash", line_color="yellow",
                                annotation_text="Compensated (true)")
                fig_v.add_trace(go.Scatter(
                    x=[t_wp], y=[v_at_wp], mode='markers',
                    marker=dict(size=12, color='yellow', symbol='x'),
                    name="Compensated (true)"))
                fig_v.add_hline(y=tuner.v_max_real, line_dash="dot", line_color="gray",
                                annotation_text=f"v_max = {tuner.v_max_real:.4f}")
            fig_v.update_layout(title="DT1: Filtered Slope v(t)",
                                xaxis_title="Time [s]",
                                yaxis_title="Slope [units/s]",
                                template="plotly_dark", height=350, hovermode="x unified")
            st.plotly_chart(fig_v, use_container_width=True)

            # === Plot 3: DT2 - Filtered Acceleration a(t) ===
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=t_arr, y=a_arr,
                                       name="a(t)", line=dict(color='#EF553B', width=2)))
            if t_wp is not None:
                # Measured (delayed)
                fig_a.add_vline(x=t_meas, line_dash="dot", line_color="red",
                                annotation_text="Measured (delayed)")
                fig_a.add_trace(go.Scatter(
                    x=[t_meas], y=[a_at_meas], mode='markers',
                    marker=dict(size=10, color='red', symbol='circle'),
                    name="Measured (delayed)"))
                # Compensated (true)
                fig_a.add_vline(x=t_wp, line_dash="dash", line_color="yellow",
                                annotation_text="Compensated (true)")
                fig_a.add_trace(go.Scatter(
                    x=[t_wp], y=[a_at_wp], mode='markers',
                    marker=dict(size=12, color='yellow', symbol='x'),
                    name="Compensated (true)"))
                fig_a.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_a.update_layout(title="DT2: Filtered Acceleration a(t)",
                                xaxis_title="Time [s]",
                                yaxis_title="Acceleration [units/s²]",
                                template="plotly_dark", height=350, hovermode="x unified")
            st.plotly_chart(fig_a, use_container_width=True)

            # === Result Boxes ===
            if tuner.tuning_done:
                st.markdown("---")

                # Box 1: Inflection Point Data
                st.markdown("### Identified Inflection Point")
                w1, w2, w3 = st.columns(3)
                w1.metric("Time t_wp", f"{tuner.t_wp_real:.2f} s")
                w2.metric("Temperature x_wp", f"{tuner.x_wp_real:.2f}")
                w3.metric("Max. Slope v_max", f"{tuner.v_max_real:.4f} /s")

                st.markdown("---")

                # Box 2: Estimated Plant Parameters
                st.markdown("### Estimated Plant Parameters")
                if tuner.plant_type == "PT2 with Dead Time":
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Dead Time Tu", f"{tuner.Tu:.2f} s")
                    s2.metric("Balance Time Tg", f"{tuner.Tg_est:.2f} s")
                    s3.metric("Gain Ks", f"{tuner.Ks_est:.3f}")
                    s4.metric("Steady-State x_max (est.)", f"{tuner.x_max_est:.2f}")
                else:
                    s1, s2 = st.columns(2)
                    s1.metric("Dead Time Tu", f"{tuner.Tu:.2f} s")
                    s2.metric("Integration Coeff. Kis", f"{tuner.Kis_est:.5f}")


    # -----------------------------------------------
    # TAB 3: PID TEST
    # -----------------------------------------------
    with tab3:
        with st.sidebar:
            st.header("3. Controller Test")

            use_tuned = st.checkbox("Use Tuned Parameters", key="use_tuned",
                                    on_change=sync_tuning_params)

            if use_tuned:
                # Auto-detect plant type from identification
                pt = st.session_state.id_plant_type
                st.caption(f"Identified Plant: **{pt}**")
                chr_mode = st.selectbox(
                    "CHR Tuning Rule",
                    ["20% Overshoot — Setpoint Tracking",
                     "20% Overshoot — Disturbance Rejection",
                     "Aperiodic — Setpoint Tracking",
                     "Aperiodic — Disturbance Rejection"],
                    key="chr_mode",
                    on_change=sync_tuning_params)

            st.markdown("---")
            rk_type = st.selectbox("Type", ["PID", "PI", "P", "PD"], key="rk_type")

            st.number_input("Kp", format="%.4f", key="rk_p", disabled=use_tuned)
            st.number_input("Tn", format="%.4f", key="rk_n", disabled=use_tuned)
            st.number_input("Tv", format="%.4f", key="rk_v", disabled=use_tuned)

            st.markdown("---")
            c_sim_time = st.slider("Duration [s]", 50, 1200, step=50, key="c_sim_time")
            c_step_val = st.number_input("Step Value", key="c_step_val")

        st.subheader("Closed-Loop Controller Test")

        # Show active CHR rule info
        if st.session_state.use_tuned and st.session_state.id_tu > 0:
            pt = st.session_state.id_plant_type
            mode = st.session_state.chr_mode
            Kp_disp, Tn_disp, Tv_disp = compute_chr_params(
                pt, st.session_state.id_tu, st.session_state.id_tg,
                st.session_state.id_ks, st.session_state.id_kis, mode)
            st.info(f"**CHR: {mode} — {pt}**:  "
                    f"Kp = {Kp_disp:.4f},  Tn = {Tn_disp:.2f} s,  Tv = {Tv_disp:.2f} s")

        if st.button("🚀 Start Simulation", type="primary"):
            p_kp = st.session_state.rk_p
            p_tn = st.session_state.rk_n
            p_tv = st.session_state.rk_v
            p_type = st.session_state.rk_type

            mp_str = json.dumps(model_params)

            t_p, x_p, y_p, w_p = simulate_pid_closedloop(
                mp_str, p_kp, p_tn, p_tv, p_type,
                st.session_state.c_step_val, st.session_state.c_sim_time,
                st.session_state.t_min, st.session_state.t_max)

            st.session_state['data_pid'] = pd.DataFrame({
                "Time": t_p, "PV": x_p, "SP": w_p, "OP": y_p
            })

            sp = st.session_state.c_step_val
            max_pv = np.max(x_p)
            overshoot = max(0.0, (max_pv - sp) / sp * 100) if sp > 0 else 0.0

            r1, r2, r3 = st.columns(3)
            r1.metric("Max. Overshoot", f"{overshoot:.2f} %", delta_color="inverse")
            r2.metric("Steady-State Value", f"{x_p[-1]:.2f}")
            r3.metric("Control Deviation", f"{sp - x_p[-1]:.4f}")

            fig_pid = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
            fig_pid.add_trace(go.Scatter(x=t_p, y=x_p, name="PV",
                                         line=dict(color='#00CC96', width=2)), row=1, col=1)
            fig_pid.add_trace(go.Scatter(x=t_p, y=w_p, name="SP",
                                         line=dict(dash='dash', color='red')), row=1, col=1)
            fig_pid.add_trace(go.Scatter(x=t_p, y=y_p, name="OP",
                                         fill='tozeroy', line=dict(color='gray')), row=2, col=1)
            fig_pid.update_layout(height=600, template="plotly_dark")
            st.plotly_chart(fig_pid, use_container_width=True)

    # -----------------------------------------------
    # TAB 4: HELP & THEORY
    # -----------------------------------------------
    with tab4:
        st.markdown(r"""
# Theory: Point-of-Inflection Autotuning

## 1. Filtered Derivatives & Compensation

Two cascaded discrete-time filters extract the slope $v$ (1st derivative) and
acceleration $a$ (2nd derivative) from the temperature signal $x$:

**1st Derivative (Slope):**
$$v_k = \frac{T_{f1}}{T_{f1} + T_a} \cdot v_{k-1} + \frac{1}{T_{f1} + T_a} \cdot (x_k - x_{k-1})$$

**2nd Derivative (Acceleration):**
$$a_k = \frac{T_{f2}}{T_{f2} + T_a} \cdot a_{k-1} + \frac{1}{T_{f2} + T_a} \cdot (v_k - v_{k-1})$$

The total filter delay is $T_{f1} + T_{f2}$. A **ring buffer** stores past measurements
to look up the true inflection point in the past.

## 2. State Machine

| State | Description |
|-------|-------------|
| **INIT** | Wait for process to settle |
| **HEATING** | Open-loop with $y_{tune}$, search for inflection point |
| **IDENTIFIED** | Inflection found, continue heating until plant settles |
| **DONE** | Parameters computed, step response complete |

## 3. Parameter Computation

### PT2 Plant (self-regulating):
- Dead time: $T_u = t_{w,real} - \frac{x_{w,real} - x_0}{v_{max,real}}$
- Steady-state estimate: $x_{max} = x_0 + \frac{x_{w,real} - x_0}{0.28}$

### IT1 Plant (integrating):
- $K_{IS} = \frac{v_{max,real}}{y_{tune}}$

## 4. CHR Tuning Rules (Chien-Hrones-Reswick)

### 20% Overshoot

| | **Setpoint Tracking** | **Disturbance Rejection** |
|---|---|---|
| **PT2** Kp | $0.95 \cdot T_g / (K_s \cdot T_u)$ | $1.2 \cdot T_g / (K_s \cdot T_u)$ |
| **PT2** Tn | $1.35 \cdot T_u$ | $2.3 \cdot T_u$ |
| **PT2** Tv | $0.47 \cdot T_u$ | $0.42 \cdot T_u$ |
| **IT1** Kp | $0.95 / (K_{IS} \cdot T_u)$ | $1.2 / (K_{IS} \cdot T_u)$ |
| **IT1** Tn | $2.0 \cdot T_u$ | $2.0 \cdot T_u$ |
| **IT1** Tv | $0.47 \cdot T_u$ | $0.42 \cdot T_u$ |

### Aperiodic (0% Overshoot)

| | **Setpoint Tracking** | **Disturbance Rejection** |
|---|---|---|
| **PT2** Kp | $0.6 \cdot T_g / (K_s \cdot T_u)$ | $0.95 \cdot T_g / (K_s \cdot T_u)$ |
| **PT2** Tn | $T_g$ | $2.4 \cdot T_u$ |
| **PT2** Tv | $0.5 \cdot T_u$ | $0.42 \cdot T_u$ |
| **IT1** Kp | $0.59 / (K_{IS} \cdot T_u)$ | $0.95 / (K_{IS} \cdot T_u)$ |
| **IT1** Tn | $2.4 \cdot T_u$ | $2.4 \cdot T_u$ |
| **IT1** Tv | $0.5 \cdot T_u$ | $0.42 \cdot T_u$ |

### Quick Guide:
1. **Tab 1:** Configure the plant model
2. **Tab 2:** Click "Start Tuning" to identify plant parameters (Tu, Tg, Ks / Kis)
3. **Tab 3:** Select CHR tuning rule, parameters are computed automatically. Click "Start Simulation" to test.
        """)

    # -----------------------------------------------
    # DATA EXPORT & CONFIG
    # -----------------------------------------------
    if any(k in st.session_state for k in ['data_model', 'data_tuning', 'data_pid']):
        frames = []
        for prefix, key in [("Model_", 'data_model'), ("Tuning_", 'data_tuning'), ("PID_", 'data_pid')]:
            if key in st.session_state:
                frames.append(st.session_state[key].copy().add_prefix(prefix))
        if frames:
            csv_data = pd.concat(frames, axis=1).to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_data, "poi_tuning_results.csv", "text/csv")

    with st.sidebar:
        st.markdown("---")
        st.header("💾 Configuration")
        keys = [
            "m_plant_type", "m_ks", "m_kis", "m_t1", "m_t2", "m_delay", "m_noise_amp",
            "t_setpoint", "t_y_tune", "t_max", "t_min", "t_tf1", "t_tf2", "t_sim_time",
            "id_tu", "id_tg", "id_ks", "id_kis", "id_plant_type",
            "use_tuned", "chr_mode", "rk_type", "rk_p", "rk_n", "rk_v", "c_sim_time", "c_step_val"
        ]
        curr_conf = {k: st.session_state[k] for k in keys if k in st.session_state}
        st.download_button("📥 Save Config", json.dumps(curr_conf, indent=2),
                           "poi_config.json", "application/json")
        st.file_uploader("📤 Load Config", type=["json"], key="uploader_key",
                         on_change=on_upload_callback)


if __name__ == "__main__":
    main()
