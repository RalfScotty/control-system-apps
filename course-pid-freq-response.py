import streamlit as st
import numpy as np
from scipy.signal import chirp as scipy_chirp, lsim, TransferFunction
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import streamlit.components.v1 as _comp

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
SYS_LABELS = ["1st Order Low-Pass", "1st Order High-Pass", "2nd Order System"]

TAU_MIN,     TAU_MAX,     TAU_STEP     = 0.01, 20.0,  0.01
XI_MIN,      XI_MAX,      XI_STEP      = 0.05,  2.0,  0.05
OMEGA_N_MIN, OMEGA_N_MAX, OMEGA_N_STEP = 0.5,  50.0,  0.5
V_MIN,       V_MAX,       V_STEP       = 0.1,   5.0,  0.1
F_TEST_MIN,  F_TEST_MAX,  F_TEST_STEP  = 0.01, 50.0,  0.01
CHIRP_F1_MIN, CHIRP_F1_MAX, CHIRP_F1_STEP = 0.01,  5.0, 0.01
CHIRP_F2_MIN, CHIRP_F2_MAX, CHIRP_F2_STEP = 1.0,  200.0, 1.0

DEFAULTS: dict = {
    "sys_label":  "1st Order Low-Pass",
    "tau":        1.0,
    "xi":         0.5,
    "omega_n":    5.0,
    "V":          1.0,
    "f_test_hz":  0.5,
    "chirp_f1":   0.1,
    "chirp_f2":   20.0,
}


def init_state() -> None:
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def safe_get(key: str):
    return st.session_state.get(key, DEFAULTS[key])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─── JSON UPLOAD CALLBACK ─────────────────────────────────────────────────────
def on_upload_callback() -> None:
    uploader = st.session_state.get("json_uploader")
    if uploader is None:
        return
    try:
        data = json.load(uploader)
    except Exception as exc:
        st.error(f"Could not read the file – is it a valid JSON file? ({exc})")
        return
    validators = {
        "sys_label":  lambda v: str(v) if str(v) in SYS_LABELS else DEFAULTS["sys_label"],
        "tau":        lambda v: clamp(float(v), TAU_MIN, TAU_MAX),
        "xi":         lambda v: clamp(float(v), XI_MIN, XI_MAX),
        "omega_n":    lambda v: clamp(float(v), OMEGA_N_MIN, OMEGA_N_MAX),
        "V":          lambda v: clamp(float(v), V_MIN, V_MAX),
        "f_test_hz":  lambda v: clamp(float(v), F_TEST_MIN, F_TEST_MAX),
        "chirp_f1":   lambda v: clamp(float(v), CHIRP_F1_MIN, CHIRP_F1_MAX),
        "chirp_f2":   lambda v: clamp(float(v), CHIRP_F2_MIN, CHIRP_F2_MAX),
    }
    for key, validator in validators.items():
        if key in data:
            try:
                st.session_state[key] = validator(data[key])
            except Exception:
                st.session_state[key] = DEFAULTS[key]
    st.toast("✅ Settings loaded!", icon="💾")


# ─── SIMULATION FUNCTIONS ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_bode(sys_label: str, tau: float, xi: float, omega_n: float, V: float) -> tuple:
    """Returns (f_hz, mag_db, phase_deg) analytically over 5 decades."""
    f_hz = np.logspace(-2, 2, 600)
    omega = 2.0 * np.pi * f_hz
    jw = 1j * omega
    if sys_label == "1st Order Low-Pass":
        H = 1.0 / (1.0 + jw * tau)
    elif sys_label == "1st Order High-Pass":
        H = (jw * tau) / (1.0 + jw * tau)
    else:
        H = V * omega_n**2 / (omega_n**2 - omega**2 + 2j * xi * omega_n * omega)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-15)
    phase_deg = np.angle(H) * 180.0 / np.pi
    return f_hz, mag_db, phase_deg


@st.cache_data(show_spinner=False)
def compute_freq_point(sys_label: str, tau: float, xi: float, omega_n: float,
                       V: float, f_test_hz: float) -> tuple:
    """Steady-state sinusoidal response at one frequency (analytical)."""
    omega = 2.0 * np.pi * f_test_hz
    jw = 1j * omega
    if sys_label == "1st Order Low-Pass":
        H_val = 1.0 / (1.0 + jw * tau)
    elif sys_label == "1st Order High-Pass":
        H_val = (jw * tau) / (1.0 + jw * tau)
    else:
        H_val = V * omega_n**2 / (omega_n**2 - omega**2 + 2j * xi * omega_n * omega)
    gain_lin  = float(np.abs(H_val))
    gain_db   = 20.0 * np.log10(max(gain_lin, 1e-15))
    phase_rad = float(np.angle(H_val))
    phase_deg = phase_rad * 180.0 / np.pi
    # 3 full cycles
    T = 1.0 / f_test_hz
    t = np.linspace(0.0, 3.0 * T, 1200)
    u = np.sin(omega * t)
    y = gain_lin * np.sin(omega * t + phase_rad)
    return t, u, y, gain_db, phase_deg, gain_lin, phase_rad


@st.cache_data(show_spinner=False)
def compute_chirp_response(sys_label: str, tau: float, xi: float, omega_n: float,
                           V: float, f1_hz: float, f2_hz: float) -> tuple:
    """System response to a logarithmic chirp signal."""
    dt = min(1.0 / (8.0 * f2_hz), 0.002)
    T_end = 25.0
    t = np.arange(0.0, T_end, dt)
    u = scipy_chirp(t, f0=f1_hz, f1=f2_hz, t1=T_end, method='logarithmic')
    if sys_label == "1st Order Low-Pass":
        num, den = [1.0], [tau, 1.0]
    elif sys_label == "1st Order High-Pass":
        num, den = [tau, 0.0], [tau, 1.0]
    else:
        num, den = [V * omega_n**2], [1.0, 2 * xi * omega_n, omega_n**2]
    _, y, _ = lsim(TransferFunction(num, den), u, t)
    return t, u, y


# ─── HTML5 CANVAS ANIMATION ───────────────────────────────────────────────────

def _render_anim(html_str: str, height: int = 295) -> None:
    """Render a canvas animation centred in the page."""
    _, col, _ = st.columns([0.03, 0.94, 0.03])
    with col:
        _comp.html(html_str, height=height)


def _anim_wrap(cid: str, w: int, h: int, js_vars: str, js_body: str) -> str:
    """Wrap a canvas + IIFE script.  js_vars uses f-string; js_body is a plain
    string so { } are safe."""
    header = (
        f'<style>body{{margin:0;padding:0;background:#0D1117}}</style>'
        f'<canvas id="{cid}" width="{w}" height="{h}" '
        f'style="display:block;margin:4px auto;border-radius:12px;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.6)"></canvas>'
        f'<script>(function(){{\n'
    )
    footer = '\n})();\n</script>'
    return header + js_vars + '\n' + js_body + footer


def bode_anim_html(f_test_hz: float, gain_db: float, gain_lin: float,
                   phase_deg: float, f_bode: np.ndarray, mag_db: np.ndarray) -> str:
    """680×270 canvas: left = dual-trace oscilloscope, right = mini Bode plot."""
    # Downsample bode arrays to 100 points
    n_bode = 100
    idx = np.round(np.linspace(0, len(f_bode) - 1, n_bode)).astype(int)
    f_j   = json.dumps([round(float(x), 5) for x in f_bode[idx]])
    mag_j = json.dumps([round(float(x), 3) for x in mag_db[idx]])

    phase_rad = phase_deg * np.pi / 180.0

    js_vars = (
        f'const F_TEST={float(f_test_hz)};\n'
        f'const GAIN_DB={float(gain_db)};\n'
        f'const GAIN_LIN={float(gain_lin)};\n'
        f'const PHASE_DEG={float(phase_deg)};\n'
        f'const PHASE_RAD={float(phase_rad)};\n'
        f'const F_BODE={f_j};\n'
        f'const MAG_BODE={mag_j};\n'
    )

    js_body = r"""
const canvas = document.getElementById('bodeA');
if (!canvas) return;
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
// Layout split
const OSC_W = 430;
const BODE_X = OSC_W + 10;
const BODE_W = W - BODE_X - 8;
// Oscilloscope tracks
const TRACK_H = H / 2;
const IN_CY   = TRACK_H * 0.5;
const OUT_CY  = TRACK_H * 1.5;
const AMP_IN  = TRACK_H * 0.32;
const AMP_OUT = AMP_IN * Math.min(GAIN_LIN, 3.5);
const SCROLL_SPD = 80; // px per second
let t0 = null;

function drawOscilloscope(elapsed) {
    // Background
    ctx.fillStyle = '#0A0F1A';
    ctx.beginPath();
    ctx.roundRect(0, 0, OSC_W, H, 10);
    ctx.fill();

    // Divider
    ctx.strokeStyle = '#1E2A3A';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, TRACK_H);
    ctx.lineTo(OSC_W, TRACK_H);
    ctx.stroke();

    // Grid lines for each track
    ctx.strokeStyle = 'rgba(40,60,90,0.5)';
    ctx.lineWidth = 0.5;
    for (let row = 0; row < 2; row++) {
        const cy = row === 0 ? IN_CY : OUT_CY;
        for (let dy = -1; dy <= 1; dy++) {
            ctx.beginPath();
            ctx.moveTo(0, cy + dy * AMP_IN * 0.7);
            ctx.lineTo(OSC_W - 60, cy + dy * AMP_IN * 0.7);
            ctx.stroke();
        }
    }

    // Scrolling offset
    const omega = 2 * Math.PI * F_TEST;
    const scroll = (elapsed * SCROLL_SPD / 1000) % OSC_W;

    // Draw input trace (blue)
    ctx.save();
    ctx.shadowColor = '#6B9FFF';
    ctx.shadowBlur = 10;
    ctx.strokeStyle = '#6B9FFF';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    let first = true;
    for (let px = 0; px < OSC_W - 60; px++) {
        const xTime = (px + scroll) / SCROLL_SPD;
        const y = IN_CY - AMP_IN * Math.sin(omega * xTime);
        if (first) { ctx.moveTo(px, y); first = false; }
        else ctx.lineTo(px, y);
    }
    ctx.stroke();
    ctx.restore();

    // Draw output trace (teal)
    ctx.save();
    ctx.shadowColor = '#00E5A0';
    ctx.shadowBlur = 12;
    ctx.strokeStyle = '#00E5A0';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    first = true;
    for (let px = 0; px < OSC_W - 60; px++) {
        const xTime = (px + scroll) / SCROLL_SPD;
        const y = OUT_CY - AMP_OUT * Math.sin(omega * xTime + PHASE_RAD);
        if (first) { ctx.moveTo(px, y); first = false; }
        else ctx.lineTo(px, y);
    }
    ctx.stroke();
    ctx.restore();

    // Amplitude arrows on right edge
    const ax = OSC_W - 58;
    // Input amplitude arrow
    ctx.strokeStyle = '#6B9FFF';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ax, IN_CY - AMP_IN);
    ctx.lineTo(ax, IN_CY + AMP_IN);
    ctx.stroke();
    ctx.fillStyle = '#6B9FFF';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('A=1', ax + 4, IN_CY - AMP_IN + 4);

    // Output amplitude arrow
    const dispAmp = Math.max(AMP_OUT, 4);
    ctx.strokeStyle = '#00E5A0';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ax, OUT_CY - dispAmp);
    ctx.lineTo(ax, OUT_CY + dispAmp);
    ctx.stroke();
    ctx.fillStyle = '#00E5A0';
    ctx.fillText('A=' + GAIN_LIN.toFixed(2), ax + 4, OUT_CY - dispAmp + 4);

    // Labels
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6B9FFF';
    ctx.fillText('Input', 8, IN_CY - AMP_IN - 10);
    ctx.fillStyle = '#00E5A0';
    const gSign = GAIN_DB >= 0 ? '+' : '';
    ctx.fillText('Output   ' + gSign + GAIN_DB.toFixed(1) + ' dB   ' + PHASE_DEG.toFixed(1) + '\u00b0', 8, OUT_CY - Math.max(AMP_OUT, 16) - 10);

    // Frequency label bottom
    ctx.fillStyle = '#556677';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('f = ' + F_TEST.toFixed(3) + ' Hz', 8, H - 8);
}

function drawBodeMini() {
    const px = BODE_X, py = 4;
    const pw = BODE_W, ph = H - 8;
    // Panel background
    ctx.fillStyle = 'rgba(10,16,26,0.95)';
    ctx.beginPath();
    ctx.roundRect(px, py, pw, ph, 10);
    ctx.fill();

    // Title
    ctx.fillStyle = '#7799BB';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Bode Magnitude', px + pw / 2, py + 16);

    // Plot area within panel
    const mx = px + 28, my = py + 26;
    const mw = pw - 36, mh = ph - 50;

    // Find dB range
    const maxDb = Math.max(...MAG_BODE, 6);
    const minDb = Math.min(...MAG_BODE, -60);
    const dbRange = maxDb - minDb || 1;

    // Log frequency mapping
    const f_min = F_BODE[0];
    const f_max = F_BODE[F_BODE.length - 1];
    const logMin = Math.log10(f_min);
    const logMax = Math.log10(f_max);

    function fToX(f) {
        return mx + (Math.log10(Math.max(f, 1e-9)) - logMin) / (logMax - logMin) * mw;
    }
    function dbToY(db) {
        return my + (1 - (db - minDb) / dbRange) * mh;
    }

    // 0 dB dashed reference
    const y0 = dbToY(0);
    ctx.save();
    ctx.strokeStyle = 'rgba(100,120,140,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(mx, y0);
    ctx.lineTo(mx + mw, y0);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    ctx.fillStyle = 'rgba(100,120,140,0.6)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0dB', mx + 2, y0 - 3);

    // Bode curve
    ctx.save();
    ctx.shadowColor = '#5599FF';
    ctx.shadowBlur = 4;
    ctx.strokeStyle = '#5599FF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let firstPt = true;
    for (let i = 0; i < F_BODE.length; i++) {
        const x = fToX(F_BODE[i]);
        const y = dbToY(MAG_BODE[i]);
        if (firstPt) { ctx.moveTo(x, y); firstPt = false; }
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // Current frequency dot (orange, glowing)
    const dotX = fToX(F_TEST);
    // Find nearest bode point for y
    let nearIdx = 0;
    let nearDist = Infinity;
    for (let i = 0; i < F_BODE.length; i++) {
        const d = Math.abs(Math.log10(F_BODE[i]) - Math.log10(F_TEST));
        if (d < nearDist) { nearDist = d; nearIdx = i; }
    }
    const dotY = dbToY(MAG_BODE[nearIdx]);
    ctx.save();
    ctx.shadowColor = '#FF9F1C';
    ctx.shadowBlur = 14;
    ctx.fillStyle = '#FF9F1C';
    ctx.beginPath();
    ctx.arc(dotX, dotY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Axes
    ctx.strokeStyle = '#2A3A4A';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(mx, my + mh);
    ctx.lineTo(mx + mw, my + mh);
    ctx.stroke();

    // x-axis labels
    ctx.fillStyle = '#556677';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    const fTicks = [0.01, 0.1, 1, 10, 100];
    for (const ft of fTicks) {
        if (ft >= f_min && ft <= f_max) {
            const tx = fToX(ft);
            ctx.fillText(ft < 1 ? ft.toFixed(2) : ft.toString(), tx, my + mh + 10);
        }
    }
    ctx.fillStyle = '#445566';
    ctx.textAlign = 'center';
    ctx.fillText('Hz', mx + mw / 2, my + mh + 22);

    // Current f + gain label
    ctx.fillStyle = '#FF9F1C';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    const lblY = Math.max(dotY - 12, my + 12);
    ctx.fillText(F_TEST.toFixed(2) + 'Hz', dotX, lblY);
    ctx.fillText(GAIN_DB.toFixed(1) + 'dB', dotX, lblY + 13);
}

function frame(ts) {
    if (!t0) t0 = ts;
    const elapsed = ts - t0;
    ctx.clearRect(0, 0, W, H);
    drawOscilloscope(elapsed);
    drawBodeMini();
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""
    return _anim_wrap('bodeA', 680, 270, js_vars, js_body)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.header("📡 Frequency Response")

        # System type – no key=, update session_state manually
        sys_label = str(safe_get("sys_label"))
        choice = st.selectbox(
            "System type",
            SYS_LABELS,
            index=SYS_LABELS.index(sys_label) if sys_label in SYS_LABELS else 0,
        )
        st.session_state["sys_label"] = choice

        st.divider()

        if choice in ("1st Order Low-Pass", "1st Order High-Pass"):
            st.slider("Time Constant τ (tau) [s]",
                      min_value=TAU_MIN, max_value=TAU_MAX, step=TAU_STEP,
                      key="tau",
                      help="Controls the cutoff frequency: f_c = 1/(2πτ)")

            st.markdown("**Quick presets:**")

            def _lp_fast():  st.session_state["tau"] = 0.1
            def _lp_slow():  st.session_state["tau"] = 2.0
            def _lp_reset(): st.session_state["tau"] = DEFAULTS["tau"]

            c1, c2, c3 = st.columns(3)
            c1.button("Fast\nτ=0.1", use_container_width=True, on_click=_lp_fast,   key="btn_lp_fast")
            c2.button("Slow\nτ=2.0", use_container_width=True, on_click=_lp_slow,   key="btn_lp_slow")
            c3.button("Reset",       use_container_width=True, on_click=_lp_reset,  key="btn_lp_reset")
        else:
            st.slider("Damping Ratio ξ (xi)",
                      min_value=XI_MIN, max_value=XI_MAX, step=XI_STEP,
                      key="xi",
                      help="< 1: resonance peak; = 1: critically damped; > 1: overdamped")
            st.slider("Natural Frequency ωn [rad/s]",
                      min_value=OMEGA_N_MIN, max_value=OMEGA_N_MAX, step=OMEGA_N_STEP,
                      key="omega_n",
                      help="The resonant frequency in radians per second")
            st.slider("DC Gain V",
                      min_value=V_MIN, max_value=V_MAX, step=V_STEP,
                      key="V",
                      help="Scales the output at very low frequencies")

            st.markdown("**Quick presets:**")

            def _2nd_under():    st.session_state["xi"] = 0.2
            def _2nd_critical(): st.session_state["xi"] = 1.0
            def _2nd_over():     st.session_state["xi"] = 1.5

            c1, c2, c3 = st.columns(3)
            c1.button("Under-\ndamped",  use_container_width=True, on_click=_2nd_under,    key="btn_2nd_under")
            c2.button("Critical",        use_container_width=True, on_click=_2nd_critical, key="btn_2nd_crit")
            c3.button("Over-\ndamped",   use_container_width=True, on_click=_2nd_over,     key="btn_2nd_over")

        st.divider()
        st.markdown("### 🎵 Single Frequency Test")
        st.slider("Test Frequency f [Hz]",
                  min_value=F_TEST_MIN, max_value=F_TEST_MAX, step=F_TEST_STEP,
                  key="f_test_hz",
                  help="The frequency of the sine wave sent through the system")

        st.divider()
        st.markdown("### 🌊 Chirp Range")
        st.slider("Chirp Start f1 [Hz]",
                  min_value=CHIRP_F1_MIN, max_value=CHIRP_F1_MAX, step=CHIRP_F1_STEP,
                  key="chirp_f1")
        st.slider("Chirp End f2 [Hz]",
                  min_value=CHIRP_F2_MIN, max_value=CHIRP_F2_MAX, step=CHIRP_F2_STEP,
                  key="chirp_f2")

        st.divider()
        st.markdown("### 💾 Settings")
        conf_data = {k: safe_get(k) for k in DEFAULTS}
        st.download_button(
            label="📥 Save settings (JSON)",
            data=json.dumps(conf_data, indent=2),
            file_name="freq_response_config.json",
            mime="application/json",
            use_container_width=True,
        )
        st.file_uploader(
            "📤 Load settings (JSON)",
            type=["json"],
            key="json_uploader",
            on_change=on_upload_callback,
        )


# ─── TAB 1: SINGLE FREQUENCY ──────────────────────────────────────────────────

def render_single_freq_tab() -> None:
    st.header("🎵 Single Frequency Test")
    st.caption(
        "Set a test frequency with the slider. "
        "Watch how much the system amplifies (or attenuates) the signal "
        "and how much it delays it."
    )

    sys_label = str(safe_get("sys_label"))
    tau       = float(safe_get("tau"))
    xi        = float(safe_get("xi"))
    omega_n   = float(safe_get("omega_n"))
    V         = float(safe_get("V"))
    f_test    = float(safe_get("f_test_hz"))

    t, u, y, gain_db, phase_deg, gain_lin, phase_rad = compute_freq_point(
        sys_label, tau, xi, omega_n, V, f_test
    )
    f_bode, mag_db, phase_bode = compute_bode(sys_label, tau, xi, omega_n, V)

    # ── Animation ──
    _render_anim(
        bode_anim_html(f_test, gain_db, gain_lin, phase_deg, f_bode, mag_db),
        height=295,
    )

    st.markdown("---")

    # ── 3 metrics ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Gain (dB)",
              f"{gain_db:.2f} dB",
              help="Positive = amplified, negative = attenuated, 0 dB = unchanged.")
    c2.metric("Phase shift",
              f"{phase_deg:.1f}°",
              help="Negative = output lags the input (delayed).")
    c3.metric("Gain (linear ratio)",
              f"{gain_lin:.4f}",
              help="Output amplitude divided by input amplitude. 1.0 = no change.")

    # ── Status callout ──
    if gain_db > -1.0:
        st.success(
            f"At {f_test:.3f} Hz the signal passes through almost unchanged "
            f"({gain_db:.1f} dB — less than 1 dB of change)."
        )
    elif gain_db > -20.0:
        st.warning(
            f"At {f_test:.3f} Hz the signal is attenuated to "
            f"{gain_lin*100:.1f}% of its original amplitude ({gain_db:.1f} dB)."
        )
    else:
        st.error(
            f"At {f_test:.3f} Hz the signal is strongly blocked — "
            f"only {gain_lin*100:.2f}% of the amplitude gets through ({gain_db:.1f} dB)."
        )

    st.markdown("---")

    # ── Two-panel Plotly chart ──
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Input & Output Waveforms (3 periods)", "Bode Magnitude — current frequency"),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.08,
    )

    # Time plot
    fig.add_trace(
        go.Scatter(x=t, y=u, mode="lines", name="Input  u(t)",
                   line=dict(color="#6B9FFF", width=2.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", name="Output  y(t)",
                   line=dict(color="#00E5A0", width=2.5)),
        row=1, col=1,
    )

    # Bode magnitude with current freq dot
    fig.add_trace(
        go.Scatter(x=f_bode, y=mag_db, mode="lines", name="Bode magnitude",
                   line=dict(color="#5599FF", width=2)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[f_test], y=[gain_db], mode="markers", name="Current freq",
                   marker=dict(color="#FF9F1C", size=14, symbol="circle",
                               line=dict(color="#FFCC44", width=2))),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1,
                     showgrid=True, gridcolor="#222", zeroline=True, zerolinecolor="#444")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1,
                     showgrid=True, gridcolor="#222", zeroline=True, zerolinecolor="#444")
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2,
                     showgrid=True, gridcolor="#222")
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2,
                     showgrid=True, gridcolor="#222",
                     zeroline=True, zerolinecolor="#445566")
    fig.add_hline(y=0, line_dash="dot", line_color="#445566", line_width=1.2,
                  row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CSV download ──
    csv_rows = ["Time [s],Input u(t),Output y(t)"]
    csv_rows += [f"{ti:.5f},{ui:.6f},{yi:.6f}" for ti, ui, yi in zip(t, u, y)]
    csv_data = "\n".join(csv_rows).encode("utf-8")
    _, col_btn = st.columns([3, 1])
    with col_btn:
        st.download_button(
            label="📊 Download waveform (CSV)",
            data=csv_data,
            file_name="single_freq_waveform.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─── TAB 2: CHIRP ─────────────────────────────────────────────────────────────

def render_chirp_tab() -> None:
    st.header("🌊 Chirp Signal — All Frequencies at Once")
    st.caption(
        "A chirp signal starts at a low frequency and sweeps to a high frequency. "
        "By sending it through the system, we can see how it responds to ALL frequencies in one go."
    )

    sys_label = str(safe_get("sys_label"))
    tau       = float(safe_get("tau"))
    xi        = float(safe_get("xi"))
    omega_n   = float(safe_get("omega_n"))
    V         = float(safe_get("V"))
    f1        = float(safe_get("chirp_f1"))
    f2        = float(safe_get("chirp_f2"))

    # Clamp f1 < f2
    if f1 >= f2:
        f2 = f1 + 1.0

    t, u, y = compute_chirp_response(sys_label, tau, xi, omega_n, V, f1, f2)

    # Subsample for plotting (2000 pts)
    n_plot = 2000
    if len(t) > n_plot:
        idx = np.round(np.linspace(0, len(t) - 1, n_plot)).astype(int)
        t_p, u_p, y_p = t[idx], u[idx], y[idx]
    else:
        t_p, u_p, y_p = t, u, y

    T_end = float(t[-1])
    # Instantaneous frequency (logarithmic chirp)
    f_inst = f1 * (f2 / f1) ** (t_p / T_end)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.40, 0.40, 0.20],
        subplot_titles=(
            "Chirp Input Signal",
            "System Output",
            "Instantaneous Frequency [Hz]",
        ),
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=t_p, y=u_p, mode="lines", name="Chirp input",
                   line=dict(color="#6B9FFF", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_p, y=y_p, mode="lines", name="System output",
                   line=dict(color="#00E5A0", width=1.5)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_p, y=f_inst, mode="lines", name="Frequency",
                   line=dict(color="#FF9F1C", width=2),
                   fill="tozeroy", fillcolor="rgba(255,159,28,0.08)"),
        row=3, col=1,
    )

    fig.update_xaxes(showgrid=True, gridcolor="#222", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#222", zeroline=True, zerolinecolor="#444")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1, type="log")

    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Notice how the output amplitude changes as the frequency sweeps — "
        "this IS the frequency response! Where the output is large, the system "
        "lets that frequency through. Where it shrinks, the system blocks it."
    )

    # ── CSV download ──
    csv_rows = ["Time [s],Input u(t),Output y(t),Freq [Hz]"]
    csv_rows += [f"{ti:.5f},{ui:.6f},{yi:.6f},{fi:.5f}"
                 for ti, ui, yi, fi in zip(t_p, u_p, y_p, f_inst)]
    csv_data = "\n".join(csv_rows).encode("utf-8")
    _, col_btn = st.columns([3, 1])
    with col_btn:
        st.download_button(
            label="📊 Download chirp data (CSV)",
            data=csv_data,
            file_name="chirp_response.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─── TAB 3: BODE PLOT ─────────────────────────────────────────────────────────

def render_bode_tab() -> None:
    st.header("📊 Bode Plot — The Complete Picture")
    st.caption(
        "The Bode plot shows how the system responds across ALL frequencies at once. "
        "The top panel shows amplitude (in dB), the bottom panel shows phase shift (in degrees)."
    )

    sys_label = str(safe_get("sys_label"))
    tau       = float(safe_get("tau"))
    xi        = float(safe_get("xi"))
    omega_n   = float(safe_get("omega_n"))
    V         = float(safe_get("V"))
    f_test    = float(safe_get("f_test_hz"))

    f_bode, mag_db, phase_deg = compute_bode(sys_label, tau, xi, omega_n, V)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Magnitude", "Phase"),
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )

    # Magnitude
    fig.add_trace(
        go.Scatter(x=f_bode, y=mag_db, mode="lines", name="Magnitude",
                   line=dict(color="#5599FF", width=2.5)),
        row=1, col=1,
    )
    # Phase
    fig.add_trace(
        go.Scatter(x=f_bode, y=phase_deg, mode="lines", name="Phase",
                   line=dict(color="#BB77FF", width=2.5)),
        row=2, col=1,
    )

    # Current test freq marker
    _, _, _, gain_db_test, phase_deg_test, _, _ = compute_freq_point(
        sys_label, tau, xi, omega_n, V, f_test
    )
    fig.add_trace(
        go.Scatter(x=[f_test], y=[gain_db_test], mode="markers",
                   name="Test freq",
                   marker=dict(color="#FF9F1C", size=12, symbol="circle",
                               line=dict(color="#FFCC44", width=2))),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[f_test], y=[phase_deg_test], mode="markers",
                   name="Test freq (phase)",
                   marker=dict(color="#FF9F1C", size=12, symbol="circle",
                               line=dict(color="#FFCC44", width=2)),
                   showlegend=False),
        row=2, col=1,
    )

    # -3 dB cutoff frequency
    if sys_label == "1st Order Low-Pass":
        f_c = 1.0 / (2.0 * np.pi * tau)
        f_c_label = f"Cutoff f_c = {f_c:.4f} Hz"
        slope_label = "-20 dB/dec"
        dc_gain_db = 0.0
    elif sys_label == "1st Order High-Pass":
        f_c = 1.0 / (2.0 * np.pi * tau)
        f_c_label = f"Cutoff f_c = {f_c:.4f} Hz"
        slope_label = "+20 dB/dec"
        dc_gain_db = -120.0  # approaches -inf
    else:
        f_c = omega_n / (2.0 * np.pi)
        f_c_label = f"Natural freq f_n = {f_c:.4f} Hz"
        slope_label = "-40 dB/dec"
        dc_gain_db = 20.0 * np.log10(max(V, 1e-15))

    # -3 dB vertical line
    fig.add_vline(x=f_c, line_dash="dash", line_color="#FF9F1C",
                  line_width=1.5, row=1, col=1)
    fig.add_vline(x=f_c, line_dash="dash", line_color="#FF9F1C",
                  line_width=1.5, row=2, col=1)
    fig.add_annotation(
        x=np.log10(f_c), y=max(mag_db) * 0.9,
        text=f_c_label, showarrow=False,
        font=dict(color="#FF9F1C", size=11),
        xref="x", yref="y",
        bgcolor="rgba(20,20,30,0.7)",
        bordercolor="#FF9F1C", borderwidth=1,
        row=1, col=1,
    )

    # 0 dB reference
    fig.add_hline(y=0, line_dash="dot", line_color="#445566", line_width=1.2,
                  row=1, col=1)
    # -180° reference
    fig.add_hline(y=-180, line_dash="dot", line_color="#445566", line_width=1.2,
                  row=2, col=1)

    # For 2nd order: annotate natural frequency separately (already done via f_c)
    if sys_label == "2nd Order System":
        fig.add_annotation(
            x=np.log10(f_c) + 0.05, y=-30,
            text=slope_label, showarrow=False,
            font=dict(color="#88AACC", size=10),
            xref="x", yref="y",
            row=1, col=1,
        )
    else:
        # Slope annotation – place in the roll-off region
        f_slope = f_c * 10
        if f_slope < f_bode[-1]:
            idx_s = np.argmin(np.abs(f_bode - f_slope))
            fig.add_annotation(
                x=np.log10(f_slope), y=mag_db[idx_s] + 8,
                text=slope_label, showarrow=False,
                font=dict(color="#88AACC", size=10),
                xref="x", yref="y",
                row=1, col=1,
            )

    fig.update_xaxes(type="log", showgrid=True, gridcolor="#222",
                     title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(type="log", showgrid=True, gridcolor="#222", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", showgrid=True, gridcolor="#222",
                     zeroline=True, zerolinecolor="#445566", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)", range=[-200, 20],
                     showgrid=True, gridcolor="#222",
                     zeroline=True, zerolinecolor="#445566", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Key metrics ──
    phase_at_cutoff = float(np.interp(np.log10(f_c),
                                      np.log10(np.maximum(f_bode, 1e-12)),
                                      phase_deg))
    c1, c2, c3 = st.columns(3)
    if sys_label == "1st Order Low-Pass":
        c1.metric("Cutoff Frequency", f"{f_c:.4f} Hz",
                  help="The -3 dB point: output is 70.7% of input here.")
    elif sys_label == "1st Order High-Pass":
        c1.metric("Cutoff Frequency", f"{f_c:.4f} Hz",
                  help="The +3 dB point (from below): output rises above 70.7% of input here.")
    else:
        c1.metric("Natural Frequency", f"{f_c:.4f} Hz",
                  help="The resonant frequency of the 2nd order system.")
    c2.metric("Phase at cutoff/nat. freq", f"{phase_at_cutoff:.1f}°",
              help="Phase shift at the key frequency. LP: ~-45°, HP: ~+45°, 2nd order: ~-90°")
    if sys_label == "2nd Order System":
        c3.metric("DC Gain", f"{dc_gain_db:.2f} dB",
                  help="Gain at very low frequencies (f → 0).")
    else:
        c3.metric("DC Gain (f → 0)", "0.00 dB" if sys_label == "1st Order Low-Pass" else "−∞ dB",
                  help="Low-pass: passes DC unchanged. High-pass: blocks DC completely.")


# ─── TAB 4: EXPLANATION ───────────────────────────────────────────────────────

def render_explanation_tab() -> None:
    st.header("📘 Understanding Frequency Response")

    st.markdown("""
> **Key idea:** Real-world systems do not treat all frequencies equally.
> A speaker plays bass notes and treble notes differently.
> A car suspension absorbs small bumps but shakes at certain speeds.
> The **Bode plot** is a tool that shows, for every frequency, how much a signal
> is amplified or blocked — and by how much it is delayed.
    """)

    st.divider()

    st.subheader("What is a Bode plot?")
    st.markdown("""
Imagine the **equalizer on a music player** — the graphic equalizer with sliders for
bass, mid, and treble. Each slider adjusts how loud that frequency range is.
If you pull the bass slider down, low-frequency sounds get quieter.
If you pull the treble slider up, high-frequency sounds get louder.

A Bode plot is exactly that equalizer picture, but drawn as a graph.
It shows:

- **Top panel (Magnitude)** — how much the system amplifies each frequency.
  - 0 dB: signal passes through unchanged (gain = 1).
  - +6 dB: signal is doubled in amplitude (gain = 2).
  - -6 dB: signal is halved (gain = 0.5).
  - -20 dB: signal is reduced to 10% of original.

- **Bottom panel (Phase)** — how much the signal is delayed at each frequency.
  - 0°: output is perfectly in sync with input.
  - -90°: output lags a quarter of a cycle behind.
  - -180°: output is completely inverted (upside down).
    """)

    st.divider()

    st.subheader("The -3 dB Cutoff Frequency")
    st.markdown("""
The **cutoff frequency** (also called the -3 dB frequency or corner frequency) is
the frequency where the output power drops to half the input power.
In terms of amplitude, this corresponds to the output being about **70.7% of the input**.

For a 1st order low-pass with time constant τ:

    f_c = 1 / (2π × τ)

**Example:** τ = 1 s → f_c ≈ 0.159 Hz.
Signals below this frequency pass through easily; signals above it are progressively blocked.

The phase shift at the cutoff frequency is exactly **-45°** for a 1st order low-pass.
    """)

    st.divider()

    st.subheader("1st Order Systems")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Low-Pass Filter**

- Lets slow (low-frequency) signals through.
- Blocks fast (high-frequency) signals.
- The transition is gradual: -20 dB per decade.

**Everyday examples:**
- A heating system responding to temperature changes (slow = yes, fast vibrations = no)
- A heavy tank filling with liquid (fast pressure pulses get smoothed out)
- The RC circuit charging through a resistor
        """)
    with col_b:
        st.markdown("""
**High-Pass Filter**

- Blocks slow (DC/low-frequency) signals.
- Lets fast (high-frequency) signals through.
- The transition is the mirror image of the low-pass.

**Everyday examples:**
- A microphone that ignores slow pressure drift but picks up sound
- A capacitor in series with a circuit (blocks DC, passes AC)
- A differentiator circuit
        """)

    st.divider()

    st.subheader("2nd Order Systems and Resonance")
    st.markdown("""
A 2nd order system has a **natural frequency ωn** and a **damping ratio ξ**.
The damping ratio determines the shape of the Bode plot:

| Damping ratio ξ | Behaviour |
|---|---|
| ξ < 1 (underdamped) | A resonance **peak** appears near ωn — the system amplifies those frequencies |
| ξ = 1 (critically damped) | Smooth roll-off, no peak |
| ξ > 1 (overdamped) | Very smooth roll-off, even softer than critically damped |

**Resonance** is why wine glasses shatter at the right pitch, bridges can shake
catastrophically in wind, and mechanical systems can vibrate violently at certain speeds.
The system stores energy and releases it at its natural frequency.

The slope after the corner frequency is **-40 dB per decade** (twice as steep as 1st order).
The phase swings from 0° to -180° (a 1st order only goes to -90°).
    """)

    st.divider()

    st.subheader("Summary: Comparing the Three Systems")
    st.markdown("""
| System | Passes | Blocks | Slope | Phase range |
|---|---|---|---|---|
| 1st Order Low-Pass | Low frequencies | High frequencies | -20 dB/dec | 0° to -90° |
| 1st Order High-Pass | High frequencies | Low frequencies | +20 dB/dec | +90° to 0° |
| 2nd Order System | Depends on ξ and ωn | — | -40 dB/dec | 0° to -180° |
    """)


# ─── TAB 5: EXERCISES ─────────────────────────────────────────────────────────

def render_exercises_tab() -> None:
    st.header("📝 Exercises")
    st.markdown(
        "Use the sidebar controls to set parameters and explore each exercise. "
        "The tabs update in real time."
    )

    with st.expander("Exercise 1 — Find the Cutoff Frequency", expanded=True):
        st.markdown("""
**Goal:** Verify the formula f_c = 1 / (2πτ) for a low-pass filter.

**Setup:**
- System: **1st Order Low-Pass**
- τ = **1.0 s**  (default)

**Predicted cutoff:** f_c = 1 / (2π × 1.0) ≈ **0.159 Hz**

**Steps:**
1. Go to the **Bode Plot** tab (Tab 3).
2. Look at the magnitude curve — find where it crosses the **-3 dB line**.
3. The vertical dashed orange line shows the calculated cutoff frequency.
4. Does it match your prediction?

**Try it:** Change τ to 0.1 s. The new cutoff should be ≈ 1.59 Hz.
Then try τ = 5.0 s. The cutoff should be ≈ 0.032 Hz.

→ *A larger time constant means a lower cutoff — the filter is "slower".*
        """)

    with st.expander("Exercise 2 — Compare Low-Pass and High-Pass"):
        st.markdown("""
**Goal:** Understand the opposite behaviour of LP and HP filters.

**Setup:** τ = **1.0 s**, f_test = **0.5 Hz** (above the cutoff at ~0.159 Hz)

**Steps:**
1. Set System = **1st Order Low-Pass**, go to Tab 1.
   Note the gain and phase at 0.5 Hz.
2. Switch to **1st Order High-Pass** (same τ).
   Note the gain and phase at the same frequency.

**What you should see:**
- At 0.5 Hz, the LP passes it well (close to 0 dB).
- The HP also passes it well at 0.5 Hz — because 0.5 Hz is *above* its cutoff.
- Now try f_test = **0.02 Hz** (well below the cutoff):
  - LP: almost 0 dB (passes DC-like signals)
  - HP: very negative dB (blocks slow signals)

→ *LP and HP are mirrors of each other, sharing the same cutoff frequency.*
        """)

    with st.expander("Exercise 3 — 2nd Order Resonance"):
        st.markdown("""
**Goal:** See the resonance peak for an underdamped 2nd order system.

**Setup:**
- System: **2nd Order System**
- ξ = **0.1** (strongly underdamped, use the "Underdamped" preset then fine-tune)
- ωn = **5.0 rad/s** → natural frequency f_n ≈ 0.796 Hz

**Steps:**
1. Set ξ = 0.1 using the slider.
2. Go to the **Bode Plot** tab.
3. Look for the large **peak** near f_n ≈ 0.796 Hz.

**Questions:**
- How many dB above 0 dB is the peak?
- Now increase ξ to 0.5, 1.0, then 1.5. What happens to the peak?
- At ξ = 1.0, the peak disappears.

→ *A low damping ratio creates a strong resonance peak — the system amplifies signals at its natural frequency.*
        """)

    with st.expander("Exercise 4 — Phase at the Cutoff Frequency"):
        st.markdown("""
**Goal:** Confirm that a 1st order low-pass has exactly -45° phase shift at f_c.

**Setup:**
- System: **1st Order Low-Pass**, τ = **1.0 s**
- f_c ≈ **0.159 Hz**

**Steps:**
1. Set f_test = **0.159 Hz** using the slider (as close as possible).
2. Go to **Tab 1** and read the **Phase** metric.
3. It should read close to **-45.0°**.

**Why -45°?** At the cutoff frequency the real and imaginary parts of H(jω) are equal,
producing a 45° angle. This is a fundamental property of 1st order systems.

**Try with High-Pass:** Same τ, same f_c.
At f_c the HP phase should be **+45°** (it leads instead of lagging).
        """)

    with st.expander("Exercise 5 — Build a Mental Bode Plot"):
        st.markdown("""
**Goal:** Predict the Bode plot shape before running the simulation.

**Setup:** System = **1st Order Low-Pass**, τ = **2.0 s**

**Your prediction:**
1. Calculate f_c = 1 / (2π × 2.0) ≈ _____ Hz
2. Below f_c: magnitude ≈ _____ dB
3. Above f_c: magnitude falls at _____ dB per decade
4. Phase at f_c: _____ degrees
5. Phase at very high frequency: _____ degrees

**Then check** by looking at the Bode Plot tab.

**Expected answers:**
- f_c ≈ 0.080 Hz
- Below f_c: ~0 dB
- Above f_c: -20 dB/decade
- Phase at f_c: -45°
- Phase at high frequency: -90°

→ *Once you can predict the shape, you truly understand the system.*
        """)

    with st.expander("Exercise 6 — Chirp vs Single Frequency"):
        st.markdown("""
**Goal:** Connect what you see in Tab 1 (single frequency) with Tab 2 (chirp).

**Setup:** System = **1st Order Low-Pass**, τ = **1.0 s**

**Part A — Single Frequency (Tab 1):**
- Set f_test = 0.05 Hz (well below cutoff). Note: output amplitude ≈ input amplitude.
- Set f_test = 0.159 Hz (at cutoff). Output should be ~70% of input.
- Set f_test = 1.59 Hz (one decade above cutoff). Output should be ~10% of input.

**Part B — Chirp (Tab 2):**
- Set Chirp Start = **0.05 Hz**, Chirp End = **5 Hz**.
- Look at the output signal: it starts large and gradually shrinks.
- The point where it reaches ~70% of the input amplitude corresponds to your f_c.

**Compare:** The chirp output shape in Tab 2 is a visual picture of the Bode magnitude curve.

→ *A chirp is like running all your single-frequency tests back to back in one sweep.*
        """)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    init_state()

    st.title("📡 Frequency Response & Bode Plots")
    st.markdown(
        "How do systems treat different frequencies? "
        "A low-pass filter blocks high-frequency noise but passes slow signals. "
        "A 2nd order system can resonate and amplify at its natural frequency. "
        "The **Bode plot** captures all of this in a single picture."
    )

    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎵 Single Frequency",
        "🌊 Chirp Signal",
        "📊 Bode Plot",
        "📘 Explanation",
        "📝 Exercises",
    ])

    with tab1:
        render_single_freq_tab()
    with tab2:
        render_chirp_tab()
    with tab3:
        render_bode_tab()
    with tab4:
        render_explanation_tab()
    with tab5:
        render_exercises_tab()


if __name__ == "__main__":
    main()
