import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import streamlit.components.v1 as _comp

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dynamic Systems", layout="wide", page_icon="⚙️")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size:16px !important; font-weight:bold !important; }
    .stMetric { background:#0E1117; padding:12px; border-radius:8px; border:1px solid #303030; }
    section[data-testid="stSidebar"] { min-width: 310px; max-width: 310px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
G = 9.81

RC_T_END, RC_DT       = 50.0,  0.05
SMD_T_END, SMD_DT     = 30.0,  0.02
HT_T_END, HT_DT_SIM   = 300.0, 0.5

RC_R_MIN, RC_R_MAX, RC_R_STEP     = 0.5,  20.0, 0.5
RC_C_MIN, RC_C_MAX, RC_C_STEP     = 0.1,  10.0, 0.1
RC_U_MIN, RC_U_MAX                = -10.0, 10.0

SMD_M_MIN, SMD_M_MAX, SMD_M_STEP  = 0.5,  20.0, 0.5
SMD_K_MIN, SMD_K_MAX, SMD_K_STEP  = 0.5,  50.0, 0.5
SMD_C_MIN, SMD_C_MAX, SMD_C_STEP  = 0.0,  30.0, 0.5
SMD_F_MIN, SMD_F_MAX, SMD_F_STEP  = 1.0,  50.0, 1.0
SMD_X0_MIN, SMD_X0_MAX            = -2.0,  2.0

HT_DT_MIN, HT_DT_MAX, HT_DT_STEP = 0.10,  0.50, 0.01
HT_DO_MIN, HT_DO_MAX, HT_DO_STEP  = 0.010, 0.10, 0.005
HT_H0_MIN, HT_H0_MAX              = 0.0,   2.0
HT_Q0_MIN, HT_Q0_MAX, HT_Q0_STEP  = 0.0, 100.0, 1.0

DEFAULTS: dict = {
    "rc_R": 5.0, "rc_C": 2.0, "rc_u0": 0.0, "rc_u_in": 5.0,
    "smd_m": 2.0, "smd_k": 8.0, "smd_c": 3.0, "smd_F": 10.0, "smd_x0": 0.0,
    "ht_d_tank": 0.30, "ht_d_out": 0.04, "ht_h0": 0.0, "ht_q0": 20.0,
}


def init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
        st.error(f"Invalid JSON file ({exc})")
        return
    validators = {
        "rc_R":      lambda v: clamp(float(v), RC_R_MIN, RC_R_MAX),
        "rc_C":      lambda v: clamp(float(v), RC_C_MIN, RC_C_MAX),
        "rc_u0":     lambda v: clamp(float(v), RC_U_MIN, RC_U_MAX),
        "rc_u_in":   lambda v: clamp(float(v), RC_U_MIN, RC_U_MAX),
        "smd_m":     lambda v: clamp(float(v), SMD_M_MIN, SMD_M_MAX),
        "smd_k":     lambda v: clamp(float(v), SMD_K_MIN, SMD_K_MAX),
        "smd_c":     lambda v: clamp(float(v), SMD_C_MIN, SMD_C_MAX),
        "smd_F":     lambda v: clamp(float(v), SMD_F_MIN, SMD_F_MAX),
        "smd_x0":    lambda v: clamp(float(v), SMD_X0_MIN, SMD_X0_MAX),
        "ht_d_tank": lambda v: clamp(float(v), HT_DT_MIN, HT_DT_MAX),
        "ht_d_out":  lambda v: clamp(float(v), HT_DO_MIN, HT_DO_MAX),
        "ht_h0":     lambda v: clamp(float(v), HT_H0_MIN, HT_H0_MAX),
        "ht_q0":     lambda v: clamp(float(v), HT_Q0_MIN, HT_Q0_MAX),
    }
    for key, validator in validators.items():
        if key in data:
            try:
                st.session_state[key] = validator(data[key])
            except Exception:
                st.session_state[key] = DEFAULTS[key]
    st.toast("✅ Settings loaded!", icon="💾")


# ─── SIMULATION ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_rc(R: float, C: float, u0: float, u_in: float) -> tuple:
    tau = R * C
    t   = np.linspace(0.0, RC_T_END, int(RC_T_END / RC_DT) + 1)

    def ode(y, _):
        return [(u_in - y[0]) / tau]

    uc = odeint(ode, [u0], t).flatten()
    return t, uc, tau


@st.cache_data(show_spinner=False)
def compute_smd(m: float, k: float, c: float, F: float, x0: float) -> tuple:
    t = np.linspace(0.0, SMD_T_END, int(SMD_T_END / SMD_DT) + 1)

    def ode(y, _):
        x, xd = y
        return [xd, (F - c * xd - k * x) / m]

    sol      = odeint(ode, [x0, 0.0], t)
    x_pos    = sol[:, 0]
    x_vel    = sol[:, 1]
    f_spring = k * x_pos
    xi       = c / (2.0 * np.sqrt(m * k))
    x_ss     = F / k
    return t, x_pos, x_vel, f_spring, xi, x_ss


@st.cache_data(show_spinner=False)
def compute_hydraulic(d_tank: float, d_out: float, h0: float, q0_lpm: float) -> tuple:
    A_tank = np.pi * (d_tank / 2.0) ** 2
    A_out  = np.pi * (d_out  / 2.0) ** 2
    q_in   = q0_lpm / 60000.0

    t = np.linspace(0.0, HT_T_END, int(HT_T_END / HT_DT_SIM) + 1)

    def ode(h, _):
        h_s  = max(h[0], 0.0)
        return [(q_in - A_out * np.sqrt(2.0 * G * h_s)) / A_tank]

    h          = np.maximum(odeint(ode, [h0], t).flatten(), 0.0)
    q_out_lpm  = A_out * np.sqrt(2.0 * G * np.maximum(h, 1e-12)) * 60000.0
    q_in_arr   = np.full_like(t, q0_lpm)
    h_ss       = (q_in / A_out) ** 2 / (2.0 * G) if q_in > 1e-12 else 0.0
    return t, h, q_out_lpm, q_in_arr, h_ss


# ─── SCHEMATIC HELPERS ────────────────────────────────────────────────────────

def _L(x0, y0, x1, y1, color="#CCCCCC", width=2):
    return dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=width), xref="x", yref="y")


def _R(x0, y0, x1, y1, color, fill="rgba(0,0,0,0)", width=2):
    return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=width), fillcolor=fill, xref="x", yref="y")


def _C(cx, cy, r, color, fill="rgba(0,0,0,0)", width=2):
    return dict(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                line=dict(color=color, width=width), fillcolor=fill, xref="x", yref="y")


def _A(x, y, text, color="#CCCCCC", size=12, xanchor="center", yanchor="middle",
       bold=False):
    txt = f"<b>{text}</b>" if bold else text
    return dict(x=x, y=y, text=txt, font=dict(color=color, size=size),
                showarrow=False, xanchor=xanchor, yanchor=yanchor,
                xref="x", yref="y")


def _base_fig(height=300) -> tuple[go.Figure, list, list]:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(14,17,23,1)",
        plot_bgcolor="rgba(18,22,30,1)",
        xaxis=dict(range=[0, 16], showgrid=False, showticklabels=False,
                   zeroline=False, fixedrange=True),
        yaxis=dict(range=[0, 10], showgrid=False, showticklabels=False,
                   zeroline=False, fixedrange=True),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False,
    )
    return fig, [], []


def _finalize(fig, shapes, annots, traces=None):
    fig.update_layout(shapes=shapes, annotations=annots)
    if traces:
        for tr in traces:
            fig.add_trace(tr)
    return fig


# ─── RC CIRCUIT SCHEMATIC ─────────────────────────────────────────────────────

def draw_rc_schematic(R: float, C: float, u_in: float, u0: float) -> go.Figure:
    fig, sh, an = _base_fig(310)
    W = "#8899BB"
    tau = R * C

    # ── outer circuit wires ──────────────────────────────────────────────────
    # bottom wire
    sh.append(_L(2.0, 1.8, 14.0, 1.8, W, 2))
    # left wire:  gap for voltage source (y 2.8 → 7.2)
    sh += [_L(2.0, 1.8, 2.0, 2.8, W, 2), _L(2.0, 7.2, 2.0, 8.5, W, 2)]
    # top wire: gap for resistor (x 5.5 → 10.5)
    sh += [_L(2.0, 8.5, 5.5, 8.5, W, 2), _L(10.5, 8.5, 14.0, 8.5, W, 2)]
    # right wire: gap for capacitor (y 3.5 → 6.5)
    sh += [_L(14.0, 1.8, 14.0, 3.5, W, 2), _L(14.0, 6.5, 14.0, 8.5, W, 2)]

    # ── voltage source ───────────────────────────────────────────────────────
    sh.append(_C(2.0, 5.0, 2.2, "#4B7BFF", "rgba(30,60,180,0.15)", 2))
    # plus/minus symbols
    an += [
        _A(2.0, 6.1, "＋", "#6B9FFF", 15, bold=True),
        _A(2.0, 3.9, "－", "#6B9FFF", 15, bold=True),
        _A(0.3, 5.0, f"{u_in:.1f} V", "#90B8FF", 12, "left"),
        _A(2.0, 1.1, f"U_in", "#6699FF", 11),
    ]

    # ── resistor (zig-zag symbol inside rounded rect) ─────────────────────────
    sh.append(_R(5.5, 7.9, 10.5, 9.1, "#FF9F1C", "rgba(255,159,28,0.12)", 2))
    # resistor zigzag as Scatter trace
    rx = np.linspace(5.6, 10.4, 16)
    ry = np.array([8.5, 8.5, 9.0, 8.0, 9.0, 8.0, 9.0, 8.0,
                   9.0, 8.0, 9.0, 8.0, 9.0, 8.0, 8.5, 8.5])
    an.append(_A(8.0, 9.6, f"R = {R:.1f} Ω", "#FF9F1C", 13, bold=True))

    # ── capacitor (two parallel plates) ──────────────────────────────────────
    sh += [
        _L(12.0, 3.5, 16.0, 3.5, "#00E5A0", 4),
        _L(12.0, 6.5, 16.0, 6.5, "#00E5A0", 4),
    ]
    # fill between plates = dielectric
    sh.append(_R(12.0, 3.5, 16.0, 6.5, "#00E5A0", "rgba(0,229,160,0.07)", 0))
    an += [
        _A(15.5, 5.0, f"C = {C:.1f} F", "#00E5A0", 13, "left", bold=True),
        _A(14.0, 2.7, "U_C(t)", "#00E5A0", 12),
    ]

    # ── junction dots ─────────────────────────────────────────────────────────
    for cx, cy in [(2.0, 1.8), (2.0, 8.5), (14.0, 1.8), (14.0, 8.5)]:
        sh.append(_C(cx, cy, 0.18, W, W, 1))

    # ── info bar ──────────────────────────────────────────────────────────────
    an.append(_A(8.0, 0.5, f"τ = R · C = {tau:.1f} s  |  U₀ = {u0:.1f} V  →  U_ss = {u_in:.1f} V",
                 "#667799", 11))

    traces = [
        go.Scatter(x=rx, y=ry, mode="lines",
                   line=dict(color="#FF9F1C", width=2.5), showlegend=False,
                   hoverinfo="skip"),
    ]
    return _finalize(fig, sh, an, traces)


# ─── SPRING-MASS-DAMPER SCHEMATIC ─────────────────────────────────────────────

def draw_smd_schematic(m: float, k: float, c: float, F: float, xi: float) -> go.Figure:
    fig, sh, an = _base_fig(310)

    # ── wall ─────────────────────────────────────────────────────────────────
    sh.append(_R(0.3, 1.0, 1.0, 9.5, "#555566", "rgba(70,70,90,0.7)", 1))
    for yw in np.arange(1.5, 9.5, 0.7):
        sh.append(_L(0.3, yw, 1.0, yw + 0.55, "#777788", 1))
    sh.append(_L(1.0, 1.0, 1.0, 9.5, "#9999BB", 3))

    # ── spring (upper track, y=7.0) ───────────────────────────────────────────
    # spring coils via Scatter trace
    sx = np.linspace(1.0, 8.5, 60)
    sy = np.zeros_like(sx)
    sy[:3] = 7.0
    sy[-3:] = 7.0
    for i in range(3, 57):
        t_norm = (i - 3) / 54.0
        sy[i] = 7.0 + 0.55 * np.sin(t_norm * 2 * np.pi * 5)

    sh += [
        _L(1.0, 6.0, 1.0, 8.0, "#BBBBBB", 1),      # attach point on wall
    ]
    an.append(_A(4.75, 8.1, f"k = {k:.1f} N/m", "#FF9F1C", 12, bold=True))

    # ── damper (lower track, y=3.5) ───────────────────────────────────────────
    sh += [
        _L(1.0, 3.0, 1.0, 4.0, "#BBBBBB", 1),      # attach point on wall
        _L(1.0, 3.5, 3.8, 3.5, "#7788FF", 2),       # rod in
        _R(3.8, 2.8, 7.2, 4.2, "#7788FF", "rgba(100,120,255,0.15)", 2),  # cylinder
        _L(5.5, 3.1, 5.5, 3.9, "#7788FF", 3),        # piston plate
        _L(5.5, 3.5, 8.5, 3.5, "#7788FF", 2),        # rod out
    ]
    an.append(_A(5.5, 2.0, f"c = {c:.1f} Ns/m", "#7788FF", 12, bold=True))

    # ── connect spring and damper to mass ─────────────────────────────────────
    sh += [
        _L(8.5, 7.0, 8.5, 6.5, "#BBBBBB", 2),
        _L(8.5, 3.5, 8.5, 4.5, "#BBBBBB", 2),
    ]

    # ── mass block ────────────────────────────────────────────────────────────
    sh.append(_R(8.5, 4.5, 12.0, 6.5, "#00E5A0", "rgba(0,229,160,0.20)", 2))
    an.append(_A(10.25, 5.5, f"m = {m:.1f} kg", "#00E5A0", 13, bold=True))

    # ── force arrow ───────────────────────────────────────────────────────────
    sh += [
        _L(12.0, 5.5, 14.5, 5.5, "#FF4B4B", 3),
        _L(14.5, 5.5, 13.6, 5.1, "#FF4B4B", 2),
        _L(14.5, 5.5, 13.6, 5.9, "#FF4B4B", 2),
    ]
    an.append(_A(15.2, 5.5, f"F = {F:.0f} N", "#FF4B4B", 13, bold=True))

    # ── x-axis label ──────────────────────────────────────────────────────────
    sh.append(_L(8.5, 1.0, 12.0, 1.0, "#555566", 1))
    an.append(_A(10.25, 0.4, "x(t)  →  displacement", "#667788", 11))

    # ── xi badge ──────────────────────────────────────────────────────────────
    xi_color = "#FF9F1C" if xi < 0.95 else ("#00E5A0" if xi <= 1.05 else "#7788FF")
    regime   = "underdamped" if xi < 0.95 else ("critical" if xi <= 1.05 else "overdamped")
    an.append(_A(8.0, 9.5, f"ξ = {xi:.3f}  ({regime})", xi_color, 12, bold=True))

    traces = [
        go.Scatter(x=sx, y=sy, mode="lines",
                   line=dict(color="#FF9F1C", width=2.5), showlegend=False,
                   hoverinfo="skip"),
    ]
    return _finalize(fig, sh, an, traces)


# ─── HYDRAULIC TANK SCHEMATIC ─────────────────────────────────────────────────

def draw_hydraulic_schematic(d_tank: float, d_out: float,
                              q0_lpm: float, h_ss: float) -> go.Figure:
    fig, sh, an = _base_fig(310)

    # tank geometry (world coords)
    tx0, tx1, ty0, ty1 = 4.5, 11.5, 1.5, 9.0
    th = ty1 - ty0  # tank height in plot units

    # water fill level: use h_ss for steady state, clamp to tank height
    h_max_m    = 2.0   # physical max level [m] (HT_H0_MAX)
    fill_frac  = min(h_ss / h_max_m, 1.0) if h_ss > 0.0 else 0.05
    water_top  = ty0 + fill_frac * th

    # ── background ────────────────────────────────────────────────────────────
    sh.append(_R(tx0, ty0, tx1, ty1, "#223344", "rgba(10,20,35,0.6)", 0))

    # ── water body ────────────────────────────────────────────────────────────
    sh.append(_R(tx0 + 0.12, ty0, tx1 - 0.12, water_top,
                 "#1E90FF", "rgba(30,100,210,0.30)", 0))

    # ── tank walls ────────────────────────────────────────────────────────────
    sh += [
        _L(tx0, ty0, tx0, ty1, "#88AACC", 3),   # left
        _L(tx1, ty0, tx1, ty1, "#88AACC", 3),   # right
        _L(tx0, ty0, tx1, ty0, "#88AACC", 3),   # bottom
    ]

    # ── water surface (animated look) ─────────────────────────────────────────
    wx = np.linspace(tx0 + 0.12, tx1 - 0.12, 40)
    wy = water_top + 0.12 * np.sin(np.linspace(0, 2 * np.pi, 40))
    sh.append(_L(tx0 + 0.12, water_top, tx1 - 0.12, water_top, "#4DAAFF", 2))
    an.append(_A(tx1 + 0.4, water_top, f"h_ss = {h_ss:.3f} m", "#4DAAFF", 12, "left", bold=True))

    # ── water level scale ─────────────────────────────────────────────────────
    sh.append(_L(tx0 - 0.3, ty0, tx0 - 0.3, ty1, "#557799", 1))
    for frac, label in [(0.0, "0"), (0.5, "1 m"), (1.0, "2 m")]:
        yy = ty0 + frac * th
        sh.append(_L(tx0 - 0.6, yy, tx0 - 0.3, yy, "#557799", 1))
        an.append(_A(tx0 - 0.8, yy, label, "#557799", 10, "right"))

    # ── inlet pipe (top) ──────────────────────────────────────────────────────
    inlet_x = (tx0 + tx1) / 2 + 1.0
    sh += [
        _L(inlet_x, ty1, inlet_x, ty1 + 1.6, "#00CC88", 4),
        _L(inlet_x - 0.8, ty1 + 1.6, inlet_x + 0.8, ty1 + 1.6, "#00CC88", 2),
    ]
    # Arrow down
    sh += [
        _L(inlet_x, ty1 + 0.3, inlet_x - 0.35, ty1 + 0.9, "#00CC88", 2),
        _L(inlet_x, ty1 + 0.3, inlet_x + 0.35, ty1 + 0.9, "#00CC88", 2),
    ]
    an.append(_A(inlet_x, ty1 + 2.1, f"q_in = {q0_lpm:.0f} L/min", "#00CC88", 12, bold=True))

    # ── outlet pipe (bottom) ──────────────────────────────────────────────────
    out_x = (tx0 + tx1) / 2 - 1.0
    sh += [
        _L(out_x - 0.5, ty0, out_x - 0.5, ty0 - 1.5, "#FF9F1C", 3),
        _L(out_x + 0.5, ty0, out_x + 0.5, ty0 - 1.5, "#FF9F1C", 3),
        _L(out_x - 0.5, ty0 - 1.5, out_x + 0.5, ty0 - 1.5, "#FF9F1C", 2),
    ]
    # Arrow down
    sh += [
        _L(out_x, ty0 - 1.5, out_x - 0.35, ty0 - 0.8, "#FF9F1C", 2),
        _L(out_x, ty0 - 1.5, out_x + 0.35, ty0 - 0.8, "#FF9F1C", 2),
    ]
    an += [
        _A(out_x, ty0 - 2.05, f"d_out = {d_out*100:.1f} cm", "#FF9F1C", 12, bold=True),
        _A(out_x, ty0 - 2.55, "q_out(t)  Torricelli", "#CC7700", 10),
    ]

    # ── tank label ────────────────────────────────────────────────────────────
    an.append(_A((tx0 + tx1) / 2, ty1 + 0.5,
                 f"d_tank = {d_tank*100:.0f} cm", "#88AACC", 11))

    # ── wave surface trace ─────────────────────────────────────────────────────
    traces = [
        go.Scatter(x=wx, y=wy, mode="lines",
                   line=dict(color="#4DAAFF", width=1.5), showlegend=False,
                   hoverinfo="skip", fill="tozeroy",
                   fillcolor="rgba(30,100,210,0.10)"),
    ]

    fig.update_layout(
        yaxis=dict(range=[-1.5, 12], showgrid=False, showticklabels=False,
                   zeroline=False, fixedrange=True),
    )
    return _finalize(fig, sh, an, traces)


# ─── HTML5 ANIMATIONS ─────────────────────────────────────────────────────────

def _render_anim(html_str: str, height: int = 295) -> None:
    """Render a canvas animation centred in the page."""
    _, col, _ = st.columns([0.03, 0.94, 0.03])
    with col:
        _comp.html(html_str, height=height)


def _anim_wrap(cid: str, w: int, h: int, js_vars: str, js_body: str) -> str:
    """Wrap a canvas + IIFE script.  js_vars uses f-string; js_body is a plain
    string so { } are safe.  The opening (function(){ is in the f-string header
    (doubled braces) and the closing })(); is appended as a plain string."""
    header = (
        f'<style>body{{margin:0;padding:0;background:#0D1117}}</style>'
        f'<canvas id="{cid}" width="{w}" height="{h}" '
        f'style="display:block;margin:4px auto;border-radius:12px;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.6)"></canvas>'
        f'<script>(function(){{\n'
    )
    footer = '\n})();\n</script>'
    return header + js_vars + '\n' + js_body + footer


# ── RC Circuit animation ──────────────────────────────────────────────────────

def rc_anim_html(t_arr: np.ndarray, uc_arr: np.ndarray,
                 u_in: float, u0: float, tau: float, R: float, C: float) -> str:
    n   = 200
    ix  = np.round(np.linspace(0, len(t_arr) - 1, n)).astype(int)
    t_j = json.dumps([round(float(x), 3) for x in t_arr[ix]])
    u_j = json.dumps([round(float(x), 4) for x in uc_arr[ix]])

    js_vars = (
        f'const tArr={t_j};\nconst ucArr={u_j};\n'
        f'const U_IN={u_in};const U0={u0};const TAU={tau};\n'
        f'const R_V={R};const C_V={C};\n'
    )

    js_body = r"""
const canvas=document.getElementById('rcA');if(!canvas)return;
const ctx=canvas.getContext('2d');
const W=canvas.width,H=canvas.height;
const CYCLE=9000;let t0=null;
// Circuit geometry
const LX=72,RX=395,TY=66,BY=214,MX=233,CMY=140;
const batR=27,capHH=22,rw=86;
// Charge particles
const NP=9;
const pa=Array.from({length:NP},(_,i)=>i/NP);
const pLen=2*((RX-LX)+(BY-TY));
function pp(f){
  f=((f%1)+1)%1;
  const t=(RX-LX)/pLen,ri=(BY-TY)/pLen,b=(RX-LX)/pLen,l=1-t-ri-b;
  if(f<t)return{x:LX+f/t*(RX-LX),y:TY};
  const f2=f-t;if(f2<ri)return{x:RX,y:TY+f2/ri*(BY-TY)};
  const f3=f2-ri;if(f3<b)return{x:RX-f3/b*(RX-LX),y:BY};
  const f4=f3-b;return{x:LX,y:BY-f4/l*(BY-TY)};
}
function gUC(ms){const f=(ms%CYCLE)/CYCLE;return ucArr[Math.min(Math.floor(f*ucArr.length),ucArr.length-1)];}
function gT(ms){const f=(ms%CYCLE)/CYCLE;return tArr[Math.min(Math.floor(f*tArr.length),tArr.length-1)];}
function gl(x0,y0,x1,y1,c,bl,lw){
  ctx.save();ctx.shadowColor=c;ctx.shadowBlur=bl;ctx.strokeStyle=c;ctx.lineWidth=lw||2.5;
  ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();ctx.restore();
}
function frame(ts){
  if(!t0)t0=ts;
  const ms=ts-t0,uc=gUC(ms),tC=gT(ms);
  const rng=Math.abs(U_IN-U0)||0.001;
  const iF=Math.max(0,Math.min(1,(U_IN-uc)/rng));
  const cF=Math.max(0,Math.min(1,(uc-Math.min(U_IN,U0))/rng));
  ctx.fillStyle='#0D1117';ctx.fillRect(0,0,W,H);
  const lv=iF>0.03,wc=lv?'#2255AA':'#1A2A3A',wb=lv?10:0;
  // wires
  gl(LX,TY,MX-rw/2-1,TY,wc,wb);gl(MX+rw/2+1,TY,RX,TY,wc,wb);
  gl(LX,BY,RX,BY,'#1A2A3A',0);
  gl(RX,TY,RX,CMY-capHH-5,wc,wb*0.6);gl(RX,CMY+capHH+5,RX,BY,'#1A2A3A',0);
  gl(LX,TY,LX,CMY-batR,'#1A2A3A',0);gl(LX,CMY+batR,LX,BY,'#1A2A3A',0);
  // battery
  ctx.save();ctx.shadowColor='#4466EE';ctx.shadowBlur=14;
  ctx.strokeStyle='#5577EE';ctx.lineWidth=2.5;
  ctx.beginPath();ctx.arc(LX,CMY,batR,0,Math.PI*2);ctx.stroke();
  ctx.fillStyle='#88AAFF';ctx.font='bold 14px monospace';ctx.textAlign='center';
  ctx.fillText('+',LX,CMY-7);ctx.fillText('\u2212',LX,CMY+14);
  ctx.shadowBlur=0;ctx.fillStyle='#5577AA';ctx.font='10px monospace';
  ctx.fillText(U_IN.toFixed(1)+'V',LX,CMY+37);ctx.restore();
  // resistor
  const rx0=MX-rw/2;
  ctx.save();ctx.shadowColor='#FF9F1C';ctx.shadowBlur=lv?16:4;
  ctx.fillStyle='rgba(255,159,28,0.1)';ctx.fillRect(rx0,TY-11,rw,22);
  ctx.strokeStyle='#FF9F1C';ctx.lineWidth=1.5;ctx.strokeRect(rx0,TY-11,rw,22);
  ctx.beginPath();ctx.moveTo(rx0+4,TY);
  for(let i=0;i<11;i++)ctx.lineTo(rx0+8+i*7,(i%2===0?TY-7:TY+7));
  ctx.lineTo(rx0+rw-4,TY);ctx.strokeStyle='#FFB84D';ctx.stroke();
  ctx.fillStyle='#FF9F1C';ctx.font='10px monospace';ctx.textAlign='center';
  ctx.fillText(R_V.toFixed(1)+'\u03a9',MX,TY-17);ctx.restore();
  // capacitor plates + glow fill
  const cpw=46;
  ctx.save();ctx.shadowColor='#00E5A0';ctx.shadowBlur=6+cF*16;
  if(cF>0.01){
    const g=ctx.createLinearGradient(0,CMY-capHH,0,CMY+capHH);
    g.addColorStop(0,'rgba(0,229,160,0)');
    g.addColorStop(0.5,`rgba(0,229,160,${0.06+cF*0.38})`);
    g.addColorStop(1,'rgba(0,229,160,0)');
    ctx.fillStyle=g;ctx.fillRect(RX-cpw/2,CMY-capHH,cpw,capHH*2);
  }
  ctx.strokeStyle='#00E5A0';ctx.lineWidth=4;
  ctx.beginPath();ctx.moveTo(RX-cpw/2,CMY-capHH);ctx.lineTo(RX+cpw/2,CMY-capHH);ctx.stroke();
  ctx.beginPath();ctx.moveTo(RX-cpw/2,CMY+capHH);ctx.lineTo(RX+cpw/2,CMY+capHH);ctx.stroke();
  ctx.fillStyle='#00CC88';ctx.font='10px monospace';ctx.textAlign='left';
  ctx.fillText(C_V.toFixed(1)+'F',RX+cpw/2+8,CMY-4);
  ctx.fillText('U_C='+uc.toFixed(1)+'V',RX+cpw/2+8,CMY+14);ctx.restore();
  // particles
  const spd=iF*0.00007;
  for(let i=0;i<NP;i++){
    pa[i]=(pa[i]+spd)%1;const p=pp(pa[i]);const al=0.25+iF*0.75;
    ctx.save();ctx.shadowColor=`rgba(80,190,255,${al})`;ctx.shadowBlur=9;
    ctx.fillStyle=`rgba(110,215,255,${al})`;
    ctx.beginPath();ctx.arc(p.x,p.y,3.4,0,Math.PI*2);ctx.fill();ctx.restore();
  }
  // right panel – voltage bars
  const px=432,pw=W-px-8,py2=18,ph=H-28;
  ctx.fillStyle='rgba(10,16,26,0.92)';
  ctx.beginPath();ctx.roundRect(px,py2,pw,ph,10);ctx.fill();
  ctx.fillStyle='#334455';ctx.font='11px monospace';ctx.textAlign='left';
  ctx.fillText('Voltage',px+10,py2+18);
  const bh=ph-54,bpy=py2+28,bw=54;
  const cx1=px+18,cx2=px+pw-18-bw;
  const maxV=Math.max(Math.abs(U_IN),Math.abs(U0),0.5);
  // U_in
  const uiH=bh*Math.min(Math.abs(U_IN)/maxV,1);
  ctx.fillStyle='rgba(80,100,250,0.15)';ctx.fillRect(cx1,bpy,bw,bh);
  ctx.save();ctx.shadowColor='#6B9FFF';ctx.shadowBlur=8;
  ctx.fillStyle='rgba(99,110,250,0.65)';ctx.fillRect(cx1,bpy+bh-uiH,bw,uiH);ctx.restore();
  ctx.fillStyle='#8899FF';ctx.font='bold 9px monospace';ctx.textAlign='center';
  ctx.fillText('U_in',cx1+bw/2,bpy+bh+13);
  ctx.fillText(U_IN.toFixed(1)+'V',cx1+bw/2,Math.max(bpy+bh-uiH-6,bpy+12));
  // U_C
  const ucH=bh*Math.min(Math.abs(uc)/maxV,1);
  ctx.fillStyle='rgba(0,180,120,0.12)';ctx.fillRect(cx2,bpy,bw,bh);
  ctx.save();ctx.shadowColor='#00E5A0';ctx.shadowBlur=ucH>5?14:2;
  ctx.fillStyle=`rgba(0,229,160,${0.35+cF*0.5})`;ctx.fillRect(cx2,bpy+bh-ucH,bw,ucH);ctx.restore();
  ctx.fillStyle='#00E5A0';ctx.font='bold 9px monospace';ctx.textAlign='center';
  ctx.fillText('U_C',cx2+bw/2,bpy+bh+13);
  ctx.fillText(uc.toFixed(1)+'V',cx2+bw/2,Math.max(bpy+bh-ucH-6,bpy+12));
  // time
  ctx.fillStyle='#334455';ctx.font='10px monospace';ctx.textAlign='right';
  ctx.fillText('t='+tC.toFixed(1)+'s',px+pw-8,py2+ph-8);
  ctx.fillText('\u03c4='+TAU.toFixed(1)+'s',px+pw-8,py2+ph-22);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""
    return _anim_wrap('rcA', 680, 270, js_vars, js_body)


# ── Spring-Mass-Damper animation ──────────────────────────────────────────────

def smd_anim_html(t_arr: np.ndarray, x_pos: np.ndarray,
                  x_ss: float, xi: float) -> str:
    n    = 200
    ix   = np.round(np.linspace(0, len(t_arr) - 1, n)).astype(int)
    t_j  = json.dumps([round(float(x), 3) for x in t_arr[ix]])
    xp_j = json.dumps([round(float(x), 5) for x in x_pos[ix]])
    x_rng = float(max(np.max(np.abs(x_pos - x_ss)), 0.01))

    js_vars = (
        f'const tArr={t_j};\nconst xArr={xp_j};\n'
        f'const X_SS={x_ss};const X_RNG={x_rng};const XI={xi};\n'
    )

    js_body = r"""
const canvas=document.getElementById('smdA');if(!canvas)return;
const ctx=canvas.getContext('2d');
const W=canvas.width,H=canvas.height;
const CYCLE=8000;let t0=null;
// Layout
const WALL_X=68,BASE_Y=H/2,MASS_W=72,MASS_H=88;
const SPR_Y=BASE_Y-24,DMP_Y=BASE_Y+24;
const SCALE=130; // px per X_RNG unit → mass travel ±SCALE px
const REST_X=310; // canvas x of mass left-edge at x=x_ss
// Trail
const TRAIL=40;const trail=new Float32Array(TRAIL).fill(REST_X);
let trailIdx=0;
function getMassX(ms){
  const f=(ms%CYCLE)/CYCLE;
  const v=xArr[Math.min(Math.floor(f*xArr.length),xArr.length-1)];
  return REST_X+(v-X_SS)/X_RNG*SCALE;
}
function drawSpring(x0,x1,y,col,blur){
  const coils=7,seg=(x1-x0)/(coils*2+2),amp=13;
  ctx.save();ctx.shadowColor=col;ctx.shadowBlur=blur;
  ctx.strokeStyle=col;ctx.lineWidth=2.5;
  ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x0+seg,y);
  for(let i=0;i<coils*2;i++)ctx.lineTo(x0+seg+(i+1)*seg,y+(i%2===0?-amp:amp));
  ctx.lineTo(x1,y);ctx.stroke();ctx.restore();
}
function drawDamper(x0,x1,y,col,blur){
  const cylW=60,cylH=22,cylX=x0+14;
  ctx.save();ctx.shadowColor=col;ctx.shadowBlur=blur;
  ctx.strokeStyle=col;ctx.lineWidth=2;
  // wall stub + rod in
  ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(cylX,y);ctx.stroke();
  // cylinder body
  ctx.fillStyle=`rgba(80,100,220,0.14)`;ctx.fillRect(cylX,y-cylH/2,cylW,cylH);
  ctx.strokeRect(cylX,y-cylH/2,cylW,cylH);
  // piston plate (moves with mass – clamped inside cylinder)
  const pistonMaxX=cylX+cylW-8;
  const pistonX=Math.min(Math.max(x1-14,cylX+8),pistonMaxX);
  ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(pistonX,y-cylH/2+3);ctx.lineTo(pistonX,y+cylH/2-3);ctx.stroke();
  // rod out to mass
  ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cylX+cylW,y);ctx.lineTo(x1,y);ctx.stroke();
  ctx.restore();
}
function frame(ts){
  if(!t0)t0=ts;
  const ms=ts-t0,massX=getMassX(ms);
  const tC=tArr[Math.min(Math.floor((ms%CYCLE)/CYCLE*tArr.length),tArr.length-1)];
  trail[trailIdx%TRAIL]=massX;trailIdx++;
  ctx.fillStyle='#0D1117';ctx.fillRect(0,0,W,H);
  // wall
  ctx.save();ctx.fillStyle='rgba(60,65,80,0.7)';ctx.fillRect(0,40,WALL_X,H-80);
  for(let y=48;y<H-80;y+=18)ctx.strokeStyle='#444455',ctx.lineWidth=1,
    ctx.beginPath(),ctx.moveTo(0,y),ctx.lineTo(WALL_X-2,y+14),ctx.stroke();
  ctx.strokeStyle='#7788AA';ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(WALL_X,40);ctx.lineTo(WALL_X,H-40);ctx.stroke();
  ctx.restore();
  // ground line
  ctx.save();ctx.strokeStyle='#1E2E3E';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(WALL_X,H-30);ctx.lineTo(W-20,H-30);ctx.stroke();
  ctx.restore();
  // trail (fading dots)
  for(let i=0;i<TRAIL;i++){
    const ti=(trailIdx-1-i+TRAIL*10)%TRAIL;
    const tx=trail[ti]+MASS_W/2;const al=0.06*(TRAIL-i)/TRAIL;
    ctx.fillStyle=`rgba(0,220,150,${al})`;
    ctx.beginPath();ctx.arc(tx,BASE_Y,6,0,Math.PI*2);ctx.fill();
  }
  // spring + damper
  const attachX=massX;
  drawSpring(WALL_X,attachX,SPR_Y,'#FF9F1C',lv=>8);
  drawDamper(WALL_X,attachX,DMP_Y,'#7788FF',6);
  // spring label
  ctx.fillStyle='#FF9F1C';ctx.font='10px monospace';ctx.textAlign='center';
  ctx.fillText('spring',WALL_X+(attachX-WALL_X)/2,SPR_Y-22);
  ctx.fillText('damper',WALL_X+(attachX-WALL_X)/2,DMP_Y+24);
  // mass block with glow
  ctx.save();ctx.shadowColor='#00E5A0';ctx.shadowBlur=18;
  const g=ctx.createLinearGradient(massX,BASE_Y-MASS_H/2,massX+MASS_W,BASE_Y+MASS_H/2);
  g.addColorStop(0,'rgba(0,200,130,0.35)');g.addColorStop(1,'rgba(0,120,80,0.25)');
  ctx.fillStyle=g;ctx.fillRect(massX,BASE_Y-MASS_H/2,MASS_W,MASS_H);
  ctx.strokeStyle='#00E5A0';ctx.lineWidth=2;ctx.strokeRect(massX,BASE_Y-MASS_H/2,MASS_W,MASS_H);
  ctx.fillStyle='#00E5A0';ctx.font='bold 11px monospace';ctx.textAlign='center';
  ctx.fillText('m',massX+MASS_W/2,BASE_Y+5);ctx.restore();
  // force arrow
  const arrowX=massX+MASS_W;
  ctx.save();ctx.shadowColor='#FF4B4B';ctx.shadowBlur=10;
  ctx.strokeStyle='#FF4B4B';ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(arrowX+4,BASE_Y);ctx.lineTo(arrowX+46,BASE_Y);ctx.stroke();
  ctx.beginPath();ctx.moveTo(arrowX+46,BASE_Y);ctx.lineTo(arrowX+30,BASE_Y-8);ctx.lineTo(arrowX+30,BASE_Y+8);ctx.closePath();
  ctx.fillStyle='#FF4B4B';ctx.fill();ctx.restore();
  // xi badge
  const xiCol=XI<0.95?'#FF9F1C':(XI<=1.05?'#00E5A0':'#7788FF');
  const xiLbl=XI<0.95?'underdamped':(XI<=1.05?'critically damped':'overdamped');
  ctx.fillStyle=xiCol;ctx.font='bold 11px monospace';ctx.textAlign='left';
  ctx.fillText('\u03be='+XI.toFixed(3)+'  '+xiLbl,WALL_X+6,28);
  // x position indicator
  const xCur=xArr[Math.min(Math.floor((ms%CYCLE)/CYCLE*xArr.length),xArr.length-1)];
  ctx.fillStyle='#445566';ctx.font='10px monospace';ctx.textAlign='right';
  ctx.fillText('x='+xCur.toFixed(4)+'m   t='+tC.toFixed(1)+'s',W-12,H-10);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""
    return _anim_wrap('smdA', 680, 260, js_vars, js_body)


# ── Hydraulic Tank animation ──────────────────────────────────────────────────

def hydraulic_anim_html(t_arr: np.ndarray, h_arr: np.ndarray,
                         h_ss: float, q_out_arr: np.ndarray, q0_lpm: float) -> str:
    n      = 200
    ix     = np.round(np.linspace(0, len(t_arr) - 1, n)).astype(int)
    t_j    = json.dumps([round(float(x), 2) for x in t_arr[ix]])
    h_j    = json.dumps([round(float(x), 5) for x in h_arr[ix]])
    qo_j   = json.dumps([round(float(x), 3) for x in q_out_arr[ix]])
    h_disp = float(max(h_ss * 1.25, float(np.max(h_arr)), 0.05))

    js_vars = (
        f'const tArr={t_j};\nconst hArr={h_j};\nconst qoArr={qo_j};\n'
        f'const H_SS={h_ss};const H_DISP={h_disp};const Q0={q0_lpm};\n'
    )

    js_body = r"""
const canvas=document.getElementById('htA');if(!canvas)return;
const ctx=canvas.getContext('2d');
const W=canvas.width,H=canvas.height;
const CYCLE=12000;let t0=null,waveOff=0;
// Tank bounds (canvas coords)
const TX0=175,TX1=455,TY_BOT=225,TY_TOP=28;
const TH=TY_BOT-TY_TOP,TW=TX1-TX0;
// Bubbles
const NB=14;
const bub=Array.from({length:NB},(_,i)=>({
  x:TX0+10+Math.random()*(TW-20),
  y:TY_BOT-10-Math.random()*TH*0.6,
  r:1.5+Math.random()*3,
  vy:0.3+Math.random()*0.5
}));
// Drops
const NDR=6;
const drops=Array.from({length:NDR},(_,i)=>({
  x:TX0+TW*0.65+i*8-20,
  y:TY_TOP-30-i*18,
  vy:1.5+Math.random()*1
}));
function getH(ms){const f=(ms%CYCLE)/CYCLE;return hArr[Math.min(Math.floor(f*hArr.length),hArr.length-1)];}
function getQO(ms){const f=(ms%CYCLE)/CYCLE;return qoArr[Math.min(Math.floor(f*qoArr.length),qoArr.length-1)];}
function getT(ms){const f=(ms%CYCLE)/CYCLE;return tArr[Math.min(Math.floor(f*tArr.length),tArr.length-1)];}
function hToY(h){return TY_BOT-Math.min(h/H_DISP,1)*TH;}
function frame(ts){
  if(!t0)t0=ts;
  const ms=ts-t0,h=getH(ms),qo=getQO(ms),tC=getT(ms);
  waveOff=(waveOff+0.04)%(Math.PI*2);
  ctx.fillStyle='#0D1117';ctx.fillRect(0,0,W,H);
  const waterY=hToY(h);
  // h_ss dashed line
  if(H_SS>0.001){
    const hssY=hToY(H_SS);
    ctx.save();ctx.setLineDash([6,4]);ctx.strokeStyle='rgba(100,150,255,0.35)';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(TX0-18,hssY);ctx.lineTo(TX1+18,hssY);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle='rgba(100,150,255,0.6)';ctx.font='9px monospace';ctx.textAlign='right';
    ctx.fillText('h_ss',TX0-20,hssY+4);ctx.restore();
  }
  // water body gradient
  if(waterY<TY_BOT-1){
    const wg=ctx.createLinearGradient(TX0,waterY,TX0,TY_BOT);
    wg.addColorStop(0,'rgba(20,110,220,0.22)');wg.addColorStop(1,'rgba(10,60,160,0.5)');
    ctx.fillStyle=wg;ctx.fillRect(TX0+2,waterY,TW-4,TY_BOT-waterY);
  }
  // wave surface
  if(waterY<TY_BOT-2){
    ctx.save();ctx.shadowColor='rgba(80,180,255,0.6)';ctx.shadowBlur=6;
    ctx.strokeStyle='rgba(80,180,255,0.8)';ctx.lineWidth=2;
    ctx.beginPath();
    for(let x=TX0;x<=TX1;x+=3){
      const y=waterY+4*Math.sin((x-TX0)*0.08+waveOff)+2*Math.sin((x-TX0)*0.2-waveOff*1.3);
      x===TX0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();ctx.restore();
  }
  // bubbles (only if water present)
  if(h>0.01){
    for(let i=0;i<NB;i++){
      bub[i].y-=bub[i].vy;
      if(bub[i].y<waterY+4){bub[i].y=TY_BOT-6;bub[i].x=TX0+10+Math.random()*(TW-20);}
      const al=Math.min((TY_BOT-bub[i].y)/TH,1)*0.55;
      ctx.save();ctx.shadowColor=`rgba(120,200,255,${al})`;ctx.shadowBlur=4;
      ctx.strokeStyle=`rgba(150,220,255,${al})`;ctx.lineWidth=1;
      ctx.beginPath();ctx.arc(bub[i].x,bub[i].y,bub[i].r,0,Math.PI*2);ctx.stroke();
      ctx.restore();
    }
  }
  // tank walls
  ctx.save();ctx.shadowColor='#5588AA';ctx.shadowBlur=8;
  ctx.strokeStyle='#6699BB';ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(TX0,TY_TOP);ctx.lineTo(TX0,TY_BOT);ctx.lineTo(TX1,TY_BOT);ctx.lineTo(TX1,TY_TOP);ctx.stroke();
  ctx.restore();
  // level scale (left)
  ctx.fillStyle='#334455';ctx.font='9px monospace';ctx.textAlign='right';
  for(let frac=0;frac<=1;frac+=0.25){
    const sy=TY_BOT-frac*TH;const lv=(frac*H_DISP).toFixed(2);
    ctx.strokeStyle='#223344';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(TX0-14,sy);ctx.lineTo(TX0-2,sy);ctx.stroke();
    ctx.fillText(lv+'m',TX0-16,sy+4);
  }
  // current level arrow + label
  const arY=Math.max(Math.min(hToY(h),TY_BOT-2),TY_TOP+2);
  ctx.save();ctx.shadowColor='#4DAAFF';ctx.shadowBlur=10;
  ctx.strokeStyle='#4DAAFF';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(TX1+4,arY);ctx.lineTo(TX1+22,arY);ctx.stroke();
  ctx.fillStyle='#4DAAFF';ctx.font='bold 10px monospace';ctx.textAlign='left';
  ctx.fillText('h='+h.toFixed(3)+'m',TX1+26,arY+4);ctx.restore();
  // inlet pipe + drops
  const inX=TX0+TW*0.65;
  ctx.save();ctx.shadowColor='#00CC88';ctx.shadowBlur=8;
  ctx.strokeStyle='#00CC88';ctx.lineWidth=5;
  ctx.beginPath();ctx.moveTo(inX,TY_TOP);ctx.lineTo(inX,TY_TOP-35);ctx.stroke();
  ctx.lineWidth=1;ctx.strokeStyle='#007755';
  ctx.beginPath();ctx.moveTo(inX-12,TY_TOP-35);ctx.lineTo(inX+12,TY_TOP-35);ctx.stroke();
  // drops
  if(Q0>0.5){
    for(let i=0;i<NDR;i++){
      drops[i].y+=drops[i].vy;
      if(drops[i].y>waterY)drops[i].y=TY_TOP-35-i*15;
      ctx.shadowBlur=6;ctx.fillStyle='rgba(0,200,140,0.7)';
      ctx.beginPath();ctx.ellipse(drops[i].x,drops[i].y,2,3.5,0,0,Math.PI*2);ctx.fill();
    }
  }
  ctx.fillStyle='#00CC88';ctx.font='bold 9px monospace';ctx.textAlign='center';
  ctx.fillText(Q0.toFixed(0)+' L/min',inX,TY_TOP-42);ctx.restore();
  // outlet pipe + stream
  const outX=TX0+TW*0.35;
  ctx.save();ctx.shadowColor=qo>0.5?'#FF9F1C':'#334455';ctx.shadowBlur=qo>0.5?10:0;
  ctx.strokeStyle=qo>0.5?'#FF9F1C':'#334455';ctx.lineWidth=5;
  ctx.beginPath();ctx.moveTo(outX,TY_BOT);ctx.lineTo(outX,TY_BOT+32);ctx.stroke();
  if(qo>0.5){
    ctx.shadowBlur=8;ctx.lineWidth=2;
    for(let i=0;i<4;i++){
      const sy2=TY_BOT+8+i*8;const sw=2+i*1.2;
      ctx.beginPath();ctx.moveTo(outX-sw,sy2);ctx.lineTo(outX+sw,sy2+5);ctx.stroke();
    }
    ctx.fillStyle='#FF9F1C';ctx.font='bold 9px monospace';ctx.textAlign='center';
    ctx.fillText(qo.toFixed(1)+' L/min',outX,TY_BOT+52);
  }
  ctx.restore();
  // time + flow label
  ctx.fillStyle='#334455';ctx.font='10px monospace';ctx.textAlign='right';
  ctx.fillText('t='+tC.toFixed(0)+'s',W-10,H-8);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""
    return _anim_wrap('htA', 680, 290, js_vars, js_body)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚙️ Parameters")

        # ── RC Circuit ────────────────────────────────────────────────────────
        st.markdown("### ⚡ RC Circuit")

        def _rc_fast():
            st.session_state["rc_R"]   = 1.0
            st.session_state["rc_C"]   = 0.5

        def _rc_slow():
            st.session_state["rc_R"]   = 15.0
            st.session_state["rc_C"]   = 8.0

        def _rc_reset():
            for k in ("rc_R", "rc_C", "rc_u0", "rc_u_in"):
                st.session_state[k] = DEFAULTS[k]

        st.slider("Resistance R [Ω]",      RC_R_MIN, RC_R_MAX, step=RC_R_STEP, key="rc_R")
        st.slider("Capacitance C [F]",     RC_C_MIN, RC_C_MAX, step=RC_C_STEP, key="rc_C")
        st.slider("Input Voltage U_in [V]", RC_U_MIN, RC_U_MAX, step=0.5,      key="rc_u_in")
        st.slider("Initial Voltage U₀ [V]", RC_U_MIN, RC_U_MAX, step=0.5,      key="rc_u0")

        c1, c2, c3 = st.columns(3)
        c1.button("⚡ Fast",  key="rc_fast",  on_click=_rc_fast,  use_container_width=True)
        c2.button("🐢 Slow",  key="rc_slow",  on_click=_rc_slow,  use_container_width=True)
        c3.button("↺ Reset", key="rc_reset", on_click=_rc_reset, use_container_width=True)

        # ── Spring-Mass-Damper ────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔩 Spring-Mass-Damper")

        def _smd_under():
            st.session_state["smd_m"] = 2.0
            st.session_state["smd_k"] = 8.0
            st.session_state["smd_c"] = 1.0

        def _smd_crit():
            st.session_state["smd_m"] = 2.0
            st.session_state["smd_k"] = 8.0
            st.session_state["smd_c"] = float(round(2.0 * np.sqrt(2.0 * 8.0), 1))

        def _smd_over():
            st.session_state["smd_m"] = 2.0
            st.session_state["smd_k"] = 8.0
            st.session_state["smd_c"] = 20.0

        st.slider("Mass m [kg]",             SMD_M_MIN, SMD_M_MAX, step=SMD_M_STEP, key="smd_m")
        st.slider("Spring Constant k [N/m]", SMD_K_MIN, SMD_K_MAX, step=SMD_K_STEP, key="smd_k")
        st.slider("Damping c [Ns/m]",        SMD_C_MIN, SMD_C_MAX, step=SMD_C_STEP, key="smd_c")
        st.slider("Applied Force F [N]",     SMD_F_MIN, SMD_F_MAX, step=SMD_F_STEP, key="smd_F")
        st.slider("Initial Position x₀ [m]", SMD_X0_MIN, SMD_X0_MAX, step=0.1,     key="smd_x0")

        d1, d2, d3 = st.columns(3)
        d1.button("⚡ Under",  key="smd_under", on_click=_smd_under, use_container_width=True)
        d2.button("✅ Crit",   key="smd_crit",  on_click=_smd_crit,  use_container_width=True)
        d3.button("🐢 Over",   key="smd_over",  on_click=_smd_over,  use_container_width=True)

        # ── Hydraulic Tank ────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💧 Hydraulic Tank")

        def _ht_fill():
            st.session_state["ht_d_tank"] = 0.30
            st.session_state["ht_d_out"]  = 0.03
            st.session_state["ht_h0"]     = 0.0
            st.session_state["ht_q0"]     = 40.0

        def _ht_drain():
            st.session_state["ht_d_tank"] = 0.30
            st.session_state["ht_d_out"]  = 0.06
            st.session_state["ht_h0"]     = 1.0
            st.session_state["ht_q0"]     = 0.0

        def _ht_steady():
            st.session_state["ht_d_tank"] = 0.30
            st.session_state["ht_d_out"]  = 0.04
            st.session_state["ht_h0"]     = 0.0
            st.session_state["ht_q0"]     = 20.0

        st.slider("Tank Diameter d_tank [m]",  HT_DT_MIN, HT_DT_MAX, step=HT_DT_STEP, key="ht_d_tank")
        st.slider("Outlet Diameter d_out [m]", HT_DO_MIN, HT_DO_MAX, step=HT_DO_STEP, key="ht_d_out")
        st.slider("Initial Level h₀ [m]",      HT_H0_MIN, HT_H0_MAX, step=0.05,       key="ht_h0")
        st.slider("Inlet Flow q_in [L/min]",   HT_Q0_MIN, HT_Q0_MAX, step=HT_Q0_STEP, key="ht_q0")

        e1, e2, e3 = st.columns(3)
        e1.button("🌊 Fill",   key="ht_fill",   on_click=_ht_fill,   use_container_width=True)
        e2.button("🔽 Drain",  key="ht_drain",  on_click=_ht_drain,  use_container_width=True)
        e3.button("⚖️ Steady", key="ht_steady", on_click=_ht_steady, use_container_width=True)

        # ── Save / Load ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💾 Settings")
        conf_data = {k: safe_get(k) for k in DEFAULTS}
        st.download_button(
            label="📥 Save all settings (JSON)",
            data=json.dumps(conf_data, indent=2),
            file_name="dyn_systems_config.json",
            mime="application/json",
            use_container_width=True,
        )
        st.file_uploader(
            "📤 Load settings (JSON)",
            type=["json"],
            key="json_uploader",
            on_change=on_upload_callback,
        )


# ─── PLOT THEME HELPERS ───────────────────────────────────────────────────────

def _sim_layout(fig, n_rows):
    fig.update_layout(
        template="plotly_dark",
        height=200 + 160 * n_rows,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    for row in range(1, n_rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor="#252535",
                         zeroline=True, zerolinecolor="#555566", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#252535",
                         zeroline=True, zerolinecolor="#555566", row=row, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)


def _centered_chart(fig):
    """Render a Plotly chart centred with padding columns."""
    _, col_c, _ = st.columns([0.05, 0.9, 0.05])
    with col_c:
        st.plotly_chart(fig, use_container_width=True)


# ─── TAB 1: RC CIRCUIT ────────────────────────────────────────────────────────

def render_rc_tab() -> None:
    R    = float(safe_get("rc_R"))
    C    = float(safe_get("rc_C"))
    u0   = float(safe_get("rc_u0"))
    u_in = float(safe_get("rc_u_in"))

    t, uc, tau = compute_rc(R, C, u0, u_in)
    uc_ss      = u_in

    st.markdown("#### ⚡ RC Circuit – Charging & Discharging")
    st.caption(
        "When you switch on a voltage, the capacitor doesn't charge instantly. "
        "The resistor slows it down. How fast depends on **τ = R · C**."
    )

    # ── schematic ─────────────────────────────────────────────────────────────
    _centered_chart(draw_rc_schematic(R, C, u_in, u0))

    # ── live animation ────────────────────────────────────────────────────────
    _render_anim(rc_anim_html(t, uc, u_in, u0, tau, R, C), height=295)

    # ── metrics ───────────────────────────────────────────────────────────────
    idx_90 = np.argmin(np.abs(uc - (u0 + 0.9 * (uc_ss - u0)))) if abs(uc_ss - u0) > 1e-9 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Time Constant τ",    f"{tau:.2f} s",
              help="After τ s the capacitor is at 63.2 % of its final voltage.")
    c2.metric("Final Voltage U_ss", f"{uc_ss:.1f} V",
              help="Steady-state capacitor voltage = U_in.")
    c3.metric("90 % Rise Time",     f"{t[idx_90]:.1f} s",
              help="Time to reach 90 % of the final voltage.")

    # ── simulation plot ────────────────────────────────────────────────────────
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        subplot_titles=("Capacitor Voltage U_C(t)", "Input Voltage U_in(t)"),
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=t, y=uc, mode="lines", name="U_C(t)",
                             line=dict(color="#00E5A0", width=3)), row=1, col=1)
    fig.add_hline(y=uc_ss, line_dash="dot", line_color="#666677", line_width=1.5,
                  annotation_text=f"U_ss = {uc_ss:.1f} V",
                  annotation_position="bottom right", row=1, col=1)
    band = 0.05 * abs(uc_ss - u0) if abs(uc_ss - u0) > 1e-9 else 0.1
    fig.add_hrect(y0=uc_ss - band, y1=uc_ss + band,
                  fillcolor="#00E5A0", opacity=0.07, line_width=0, row=1, col=1)
    fig.add_vline(x=tau, line_dash="dash", line_color="#FF9F1C", line_width=1.5,
                  annotation_text=f"τ={tau:.1f}s",
                  annotation_position="top right", row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.full_like(t, u_in), mode="lines",
                             name="U_in(t)", line=dict(color="#6B9FFF", width=2)),
                  row=2, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    _sim_layout(fig, 2)
    _centered_chart(fig)

    # CSV
    csv = "Time [s],U_C(t) [V],U_in [V]\n" + "\n".join(
        f"{a:.3f},{b:.6f},{u_in:.3f}" for a, b in zip(t, uc))
    _, cb = st.columns([3, 1])
    with cb:
        st.download_button("📊 CSV", csv.encode(), "rc_data.csv", "text/csv",
                           use_container_width=True)


# ─── TAB 2: SPRING-MASS-DAMPER ────────────────────────────────────────────────

def render_smd_tab() -> None:
    m   = float(safe_get("smd_m"))
    k   = float(safe_get("smd_k"))
    c   = float(safe_get("smd_c"))
    F   = float(safe_get("smd_F"))
    x0  = float(safe_get("smd_x0"))

    t, x_pos, x_vel, f_spring, xi, x_ss = compute_smd(m, k, c, F, x0)

    st.markdown("#### 🔩 Spring-Mass-Damper – Mechanical Oscillation")
    st.caption(
        "Apply a force to a mass on a spring. Does it bounce, or does the damper "
        "absorb the energy? The **damping ratio ξ** decides everything."
    )

    # ── schematic ─────────────────────────────────────────────────────────────
    _centered_chart(draw_smd_schematic(m, k, c, F, xi))

    # ── live animation ────────────────────────────────────────────────────────
    _render_anim(smd_anim_html(t, x_pos, x_ss, xi), height=285)

    # ── regime badge ──────────────────────────────────────────────────────────
    overshoot = max(0.0, (float(np.max(x_pos)) - x_ss) / x_ss * 100.0) if x_ss > 1e-9 else 0.0
    band_2pct = 0.02 * abs(x_ss)
    outside   = np.where(np.abs(x_pos - x_ss) > band_2pct)[0]
    t_settle  = float(t[outside[-1]]) if len(outside) > 0 else 0.0

    if xi < 0.95:
        st.warning(f"**ξ = {xi:.3f} → Underdamped** – the mass oscillates before settling.")
    elif xi <= 1.05:
        st.success(f"**ξ = {xi:.3f} → Critically Damped** – fastest response, zero overshoot.")
    else:
        st.info(f"**ξ = {xi:.3f} → Overdamped** – no oscillation, slow approach.")

    # ── metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Damping Ratio ξ",    f"{xi:.3f}")
    c2.metric("Static Deflection",  f"{x_ss:.4f} m",
              help="Steady-state position = F / k")
    c3.metric("Overshoot",          f"{overshoot:.1f} %")

    # ── simulation plot ────────────────────────────────────────────────────────
    fig = make_subplots(rows=3, cols=1, row_heights=[0.45, 0.30, 0.25],
                        subplot_titles=("Position x(t)", "Velocity ẋ(t)", "Spring Force (N)"),
                        vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=t, y=x_pos, mode="lines", name="x(t)",
                             line=dict(color="#00E5A0", width=3)), row=1, col=1)
    fig.add_hline(y=x_ss, line_dash="dot", line_color="#666677", line_width=1.5,
                  annotation_text=f"x_ss={x_ss:.3f}m",
                  annotation_position="bottom right", row=1, col=1)
    fig.add_hrect(y0=x_ss - band_2pct, y1=x_ss + band_2pct,
                  fillcolor="#00E5A0", opacity=0.07, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_vel, mode="lines", name="ẋ(t)",
                             line=dict(color="#6B9FFF", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_spring, mode="lines", name="F_spring(t)",
                             line=dict(color="#FF9F1C", width=2)), row=3, col=1)
    fig.add_hline(y=F, line_dash="dot", line_color="#FF4B4B", line_width=1.5,
                  annotation_text=f"F={F:.0f}N",
                  annotation_position="bottom right", row=3, col=1)
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Force (N)", row=3, col=1)
    _sim_layout(fig, 3)
    _centered_chart(fig)

    csv = "Time [s],Position [m],Velocity [m/s],Spring Force [N]\n" + "\n".join(
        f"{a:.3f},{b:.6f},{c_:.6f},{d:.6f}" for a, b, c_, d in zip(t, x_pos, x_vel, f_spring))
    _, cb = st.columns([3, 1])
    with cb:
        st.download_button("📊 CSV", csv.encode(), "smd_data.csv", "text/csv",
                           use_container_width=True)


# ─── TAB 3: HYDRAULIC TANK ────────────────────────────────────────────────────

def render_hydraulic_tab() -> None:
    d_tank = float(safe_get("ht_d_tank"))
    d_out  = float(safe_get("ht_d_out"))
    h0     = float(safe_get("ht_h0"))
    q0     = float(safe_get("ht_q0"))

    t, h, q_out_lpm, q_in_arr, h_ss = compute_hydraulic(d_tank, d_out, h0, q0)

    st.markdown("#### 💧 Hydraulic Tank – Nonlinear Fluid Dynamics")
    st.caption(
        "Water flows in at a constant rate and drains through a hole. "
        "Faster drain when the level is higher (Torricelli's law) → **nonlinear** system."
    )

    A_tank     = np.pi * (d_tank / 2.0) ** 2
    A_out      = np.pi * (d_out  / 2.0) ** 2
    q_out_ss   = A_out * np.sqrt(2.0 * G * h_ss) * 60000.0

    # ── schematic ─────────────────────────────────────────────────────────────
    _centered_chart(draw_hydraulic_schematic(d_tank, d_out, q0, h_ss))

    # ── live animation ────────────────────────────────────────────────────────
    _render_anim(hydraulic_anim_html(t, h, h_ss, q_out_lpm, q0), height=315)

    # ── metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Steady-State Level h_ss",
              f"{h_ss:.3f} m" if h_ss > 0.001 else "Tank drains",
              help="Level where inflow = outflow")
    c2.metric("Peak Level",  f"{float(np.max(h)):.3f} m")
    c3.metric("Steady Outflow", f"{q_out_ss:.1f} L/min" if h_ss > 0.001 else "0 L/min")

    st.caption(
        f"A_tank = {A_tank*10000:.1f} cm²  |  "
        f"A_out = {A_out*10000:.3f} cm²  |  "
        f"Volume at h_ss ≈ {A_tank * h_ss * 1000:.1f} L"
    )

    # ── simulation plot ────────────────────────────────────────────────────────
    fig = make_subplots(rows=3, cols=1, row_heights=[0.45, 0.30, 0.25],
                        subplot_titles=("Water Level h(t)", "Outlet Flow q_out(t)",
                                        "Inlet Flow q_in(t)"),
                        vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=t, y=h, mode="lines", name="h(t)",
                             line=dict(color="#4DAAFF", width=3)), row=1, col=1)
    if h_ss > 0.001:
        fig.add_hline(y=h_ss, line_dash="dot", line_color="#666677", line_width=1.5,
                      annotation_text=f"h_ss={h_ss:.3f}m",
                      annotation_position="bottom right", row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=q_out_lpm, mode="lines", name="q_out(t)",
                             line=dict(color="#FF9F1C", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=q_in_arr, mode="lines", name="q_in(t)",
                             line=dict(color="#00CC88", width=2)), row=3, col=1)
    fig.update_yaxes(title_text="Level (m)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (L/min)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (L/min)", row=3, col=1)
    _sim_layout(fig, 3)
    _centered_chart(fig)

    csv = "Time [s],Level [m],Outflow [L/min],Inflow [L/min]\n" + "\n".join(
        f"{a:.2f},{b:.6f},{c_:.4f},{d:.3f}" for a, b, c_, d in zip(t, h, q_out_lpm, q_in_arr))
    _, cb = st.columns([3, 1])
    with cb:
        st.download_button("📊 CSV", csv.encode(), "hydraulic_data.csv", "text/csv",
                           use_container_width=True)


# ─── TAB 4: EXPLANATION ───────────────────────────────────────────────────────

def render_explanation_tab() -> None:
    st.header("📘 How Dynamic Systems Work")
    st.markdown("""
> **Key idea:** Every real system needs *time* to react. When you change the input,
> the output doesn't jump there instantly – it takes a path.
> The shape of that path tells you everything about the system's character.
    """)
    st.divider()

    col_rc, col_smd = st.columns(2)
    with col_rc:
        st.subheader("⚡ RC Circuit – 1st-Order")
        st.markdown("""
**Analogy:** Filling a bathtub.
- Resistor = narrow pipe → limits how fast charge flows in
- Capacitor = bathtub → stores the charge

The voltage rises **exponentially** and never overshoots.

| Time | Charge reached |
|---|---|
| 1τ | 63.2 % |
| 3τ | 95.0 % |
| 5τ | 99.3 % ✅ |

**ODE:** dU_C/dt = (U_in – U_C) / (R·C)
        """)

    with col_smd:
        st.subheader("🔩 Spring-Mass-Damper – 2nd-Order")
        st.markdown("""
**Analogy:** Car suspension.
- Spring → restoring force
- Damper → resists fast movement, removes energy

**Damping ratio ξ:**
- ξ < 1 → bounces (underdamped)
- ξ = 1 → fastest, no bounce (critical)
- ξ > 1 → slow, no bounce (overdamped)

**ODE:** m·x'' + c·x' + k·x = F
        """)

    st.divider()
    st.subheader("💧 Hydraulic Tank – Nonlinear 1st-Order")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Analogy:** Kitchen sink.
- Tap on = constant inflow
- Drain speed ∝ √h  ← **nonlinear!**

Doubling the level does NOT double the drain speed.

**ODE:** dh/dt = (q_in − A_out·√(2g·h)) / A_tank
        """)
    with col_b:
        st.markdown("""
**Steady state** (level constant):
h_ss = (q_in / A_out)² / (2g)

**Key insight:** h_ss ∝ 1/d_out⁴
→ Halving d_out increases h_ss by 16×!

This nonlinearity is what makes fluid systems
behave very differently from RC circuits.
        """)

    st.divider()
    st.markdown("""
| | RC Circuit | Spring-Mass-Damper | Hydraulic Tank |
|---|---|---|---|
| Order | 1st | 2nd | 1st (nonlinear) |
| Oscillation? | No | Yes if ξ < 1 | No |
| Key parameter | τ = R·C | ξ = c/(2√mk) | h_ss ∝ q_in²/d_out⁴ |
| Energy storage | Capacitor | Spring | Tank volume |
| Energy loss | Resistor | Damper | Outlet pipe |
    """)


# ─── TAB 5: EXERCISES ─────────────────────────────────────────────────────────

def render_exercises_tab() -> None:
    st.header("📝 Exercises")
    st.markdown("Adjust parameters in the **left sidebar** and watch how each system responds.")

    with st.expander("⚡ Exercise 1 – RC: Verify the Time Constant", expanded=True):
        st.markdown("""
**Setup:** R = 5 Ω, C = 2 F, U_in = 5 V, U₀ = 0 V  →  τ = 10 s

1. Read the capacitor voltage at t = 10 s in the plot
2. It should be **3.16 V** (= 63.2 % of 5 V)
3. At t = 50 s (= 5τ) it should almost equal U_in

**Challenge:** R = 10 Ω, C = 3 F  →  what is τ?
→ *30 s. Capacitor at 63 % of U_in at t = 30 s.*
        """)

    with st.expander("⚡ Exercise 2 – RC: R vs C, who matters more?"):
        st.markdown("""
**Start:** R = 5, C = 2 (τ = 10 s)

- Double R → τ = 20 s (slower)
- Reset, double C → τ = 20 s (same effect!)

**Insight:** Both R and C equally control τ. It is simply their product.

→ *τ = R·C – the circuit doesn't care which one you doubled.*
        """)

    with st.expander("🔩 Exercise 3 – SMD: Find Critical Damping"):
        st.markdown("""
**Setup:** m = 2 kg, k = 8 N/m, F = 10 N

Critical damping: c_crit = 2·√(m·k) = 2·√(16) = **8 Ns/m**

1. Set c = 1 → watch oscillation (ξ ≈ 0.18)
2. Press **✅ Crit** button → c = 8, ξ = 1.0, no overshoot
3. Set c = 20 → overdamped, sluggish

→ *Critical damping: fastest response without bouncing.*
        """)

    with st.expander("🔩 Exercise 4 – SMD: Mass vs Frequency"):
        st.markdown("""
**Setup:** k = 8 N/m, c = 1 Ns/m, F = 10 N

Natural frequency: ω_n = √(k/m)

| m [kg] | ω_n [rad/s] | Period [s] |
|---|---|---|
| 0.5 | 4.0 | 1.6 |
| 2.0 | 2.0 | 3.1 |
| 8.0 | 1.0 | 6.3 |

Count oscillation periods in the plot – do they match?

→ *Heavier mass = slower oscillation.*
        """)

    with st.expander("💧 Exercise 5 – Hydraulic: Predict Steady State"):
        st.markdown("""
**Setup:** d_tank = 0.30 m, d_out = 0.04 m, h₀ = 0 m

Formula: h_ss = (q_in / A_out)² / (2g)
where A_out = π·(0.02)² ≈ 0.001257 m²

| q_in [L/min] | Predicted h_ss |
|---|---|
| 10 | calculate! |
| 20 | 4× larger than q_in=10 result |
| 40 | 16× larger than q_in=10 result |

→ *Nonlinearity: doubling inflow **quadruples** the level.*
        """)

    with st.expander("💧 Exercise 6 – Hydraulic: Outlet Diameter Sensitivity"):
        st.markdown("""
**Setup:** d_tank = 0.30 m, h₀ = 0 m, q_in = 20 L/min

Try d_out = 0.03, 0.04, 0.05 m and note h_ss each time.

Since h_ss ∝ 1/d_out⁴:
- +25 % d_out → h_ss drops to ~41 % of original
- −25 % d_out → h_ss rises to ~316 % of original

→ *Small outlet changes have enormous level effects – this is why pipe sizing matters.*
        """)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    init_state()

    st.title("⚙️ Dynamic Systems – How Real Systems React Over Time")
    st.markdown(
        "Three classic physical systems, each responding differently to a sudden change. "
        "Adjust all parameters in the **sidebar on the left**."
    )

    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚡ RC Circuit",
        "🔩 Spring-Mass-Damper",
        "💧 Hydraulic Tank",
        "📘 Explanation",
        "📝 Exercises",
    ])

    with tab1:
        render_rc_tab()
    with tab2:
        render_smd_tab()
    with tab3:
        render_hydraulic_tab()
    with tab4:
        render_explanation_tab()
    with tab5:
        render_exercises_tab()


if __name__ == "__main__":
    main()
