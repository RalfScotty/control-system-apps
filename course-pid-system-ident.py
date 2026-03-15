import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="System Identification", layout="wide", page_icon="🔍")

st.markdown("""
<style>
    button[data-baseweb="tab"] { font-size:16px !important; font-weight:bold !important; }
    .stMetric { background:#0E1117; padding:12px; border-radius:8px; border:1px solid #303030; }
    section[data-testid="stSidebar"] { min-width: 310px; max-width: 310px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
PT1_K_MIN,  PT1_K_MAX,  PT1_K_STEP   = 0.1, 10.0, 0.1
PT1_T_MIN,  PT1_T_MAX,  PT1_T_STEP   = 0.5, 30.0, 0.5

IT1_KI_MIN, IT1_KI_MAX, IT1_KI_STEP  = 0.02, 2.0, 0.02
IT1_T_MIN,  IT1_T_MAX,  IT1_T_STEP   = 0.5, 20.0, 0.5

PT2_K_MIN,  PT2_K_MAX,  PT2_K_STEP   = 0.1, 10.0, 0.1
PT2_T1_MIN, PT2_T1_MAX, PT2_T1_STEP  = 0.5, 30.0, 0.5
PT2_T2_MIN, PT2_T2_MAX, PT2_T2_STEP  = 0.5, 30.0, 0.5

P2O_K_MIN,  P2O_K_MAX,  P2O_K_STEP   = 0.1, 5.0, 0.1
P2O_FN_MIN, P2O_FN_MAX, P2O_FN_STEP  = 0.05, 5.0, 0.05
P2O_XI_MIN, P2O_XI_MAX, P2O_XI_STEP  = 0.02, 0.95, 0.01

SYS_TYPES = [
    "PT1 – 1st Order Lag",
    "IT1 – Integrator + Lag",
    "PT2 – 2nd Order Non-Oscillating",
    "PT2osc – 2nd Order Oscillating",
]

DEFAULTS: dict = {
    "sys_type": "PT1 – 1st Order Lag",
    "pt1_K":  2.0,  "pt1_T":  5.0,
    "it1_KI": 0.3,  "it1_T":  4.0,
    "pt2_K":  1.5,  "pt2_T1": 8.0,  "pt2_T2": 2.0,
    "p2o_K":  1.0,  "p2o_fn": 0.5,  "p2o_xi": 0.3,
}

DARK = "#0D1117"
GRID = "#21262D"
TEXT = "#C9D1D9"
ACC1 = "#58A6FF"   # blue   – step response curve
ACC3 = "#F78166"   # orange – identification markers
ACC4 = "#D2A8FF"   # purple – tangent / asymptote
YELL = "#E3B341"   # yellow – final value / labels


# ─── STATE HELPERS ────────────────────────────────────────────────────────────
def init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def safe_get(key: str):
    return st.session_state.get(key, DEFAULTS[key])


_BASE = dict(
    paper_bgcolor=DARK, plot_bgcolor=DARK,
    font=dict(color=TEXT, size=13),
    margin=dict(l=65, r=160, t=55, b=70),
)
_AX   = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
_LEG  = dict(orientation="v", x=1.02, y=1.0, xanchor="left", yanchor="top",
             bgcolor=DARK, bordercolor=GRID, borderwidth=1, font=dict(size=11))


# ─── STEP RESPONSE MATH ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_pt1(K: float, T: float) -> tuple:
    t_end = max(6.0 * T, 10.0)
    t = np.linspace(0.0, t_end, 1200)
    y = K * (1.0 - np.exp(-t / T))
    return t, y


@st.cache_data(show_spinner=False)
def compute_it1(K_I: float, T: float) -> tuple:
    t_end = max(8.0 * T, 20.0)
    t = np.linspace(0.0, t_end, 1200)
    y = K_I * (t - T * (1.0 - np.exp(-t / T)))
    return t, y


@st.cache_data(show_spinner=False)
def compute_pt2(K: float, T1: float, T2: float) -> tuple:
    T1e = max(T1, T2)
    T2e = min(T1, T2)
    t_end = max(6.0 * T1e, 10.0)
    t = np.linspace(0.0, t_end, 1200)
    if abs(T1e - T2e) < 0.05:
        y = K * (1.0 - (1.0 + t / T1e) * np.exp(-t / T1e))
    else:
        y = K * (1.0 - (T1e * np.exp(-t / T1e) - T2e * np.exp(-t / T2e)) / (T1e - T2e))
    return t, y, T1e, T2e


@st.cache_data(show_spinner=False)
def compute_pt2osc(K: float, f_n: float, xi: float) -> tuple:
    wn = 2.0 * np.pi * f_n
    wd = wn * np.sqrt(1.0 - xi**2)
    T_d = 2.0 * np.pi / wd
    t_end = max(8.0 / (xi * wn), 4.0 * T_d, 2.0)
    t = np.linspace(0.0, t_end, 2000)
    y = K * (1.0 - np.exp(-xi * wn * t) * (
        np.cos(wd * t) + (xi / np.sqrt(1.0 - xi**2)) * np.sin(wd * t)
    ))
    return t, y, wn, wd, T_d


# ─── PLOT: PT1 ────────────────────────────────────────────────────────────────
def make_pt1_plot(K: float, T: float) -> go.Figure:
    t, y = compute_pt1(K, T)
    t_end = float(t[-1])

    # tangent at t=0: y = (K/T)*t  (intersects K at t=T)
    t_tang = np.array([0.0, T * 1.3])
    y_tang = (K / T) * t_tang

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Step response",
                             line=dict(color=ACC1, width=2.5)))
    fig.add_trace(go.Scatter(x=t_tang, y=y_tang, mode="lines",
                             name="Tangent at t = 0",
                             line=dict(color=ACC4, width=1.5, dash="dash")))

    y_lo = -K * 0.14
    y_hi = K * 1.20
    shapes = [
        dict(type="line", x0=0, x1=t_end, y0=K, y1=K, xref="x", yref="y",
             line=dict(color=YELL, width=1.5, dash="dash")),
        dict(type="line", x0=0, x1=T, y0=0.632 * K, y1=0.632 * K, xref="x", yref="y",
             line=dict(color=ACC3, width=1.2, dash="dot")),
        dict(type="line", x0=T, x1=T, y0=y_lo, y1=0.632 * K, xref="x", yref="y",
             line=dict(color=ACC3, width=1.2, dash="dot")),
        dict(type="line", x0=3*T, x1=3*T, y0=y_lo, y1=0.95 * K, xref="x", yref="y",
             line=dict(color=ACC3, width=1.0, dash="dot")),
        dict(type="line", x0=5*T, x1=5*T, y0=y_lo, y1=0.993 * K, xref="x", yref="y",
             line=dict(color=ACC3, width=1.0, dash="dot")),
    ]
    annots = [
        # K label – left side, above the dashed K-line
        dict(x=t_end * 0.03, y=K * 1.02, xref="x", yref="y",
             text=f"K = {K:.2f}", font=dict(color=YELL, size=11),
             showarrow=False, xanchor="left", yanchor="bottom"),
        # 63.2% label – to the right of the T-vertical, just above the horizontal line
        dict(x=T * 1.05, y=0.632 * K, xref="x", yref="y",
             text="63.2 %", font=dict(color=ACC3, size=10),
             showarrow=False, xanchor="left", yanchor="bottom"),
        # time-axis labels well below x=0
        dict(x=T,    y=y_lo * 0.75, xref="x", yref="y",
             text=f"T = {T:.1f} s", font=dict(color=ACC3, size=11),
             showarrow=False, xanchor="center", yanchor="middle"),
        dict(x=3*T,  y=y_lo * 0.75, xref="x", yref="y",
             text="3T", font=dict(color=ACC3, size=10),
             showarrow=False, xanchor="center", yanchor="middle"),
        dict(x=5*T,  y=y_lo * 0.75, xref="x", yref="y",
             text="5T", font=dict(color=ACC3, size=10),
             showarrow=False, xanchor="center", yanchor="middle"),
    ]
    fig.update_layout(
        **_BASE,
        title=dict(text="PT1 – Step Response with Identification Markers", font=dict(size=15)),
        xaxis=dict(title="Time [s]", **_AX),
        yaxis=dict(title="Output y(t)", range=[y_lo, y_hi], **_AX),
        shapes=shapes, annotations=annots,
        legend=_LEG,
        height=420,
    )
    return fig


# ─── PLOT: IT1 ────────────────────────────────────────────────────────────────
def make_it1_plot(K_I: float, T: float) -> go.Figure:
    t, y = compute_it1(K_I, T)
    t_end = float(t[-1])

    # asymptote: y_a = K_I * (t - T)
    t_asym = np.array([T, t_end])
    y_asym = K_I * (t_asym - T)

    # slope annotation: pick two points on asymptote
    t_s1, t_s2 = t_end * 0.55, t_end * 0.75
    y_s1 = K_I * (t_s1 - T)
    y_s2 = K_I * (t_s2 - T)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Step response",
                             line=dict(color=ACC1, width=2.5)))
    fig.add_trace(go.Scatter(x=t_asym, y=y_asym, mode="lines",
                             name=f"Asymptote  slope = K_I = {K_I:.3f}",
                             line=dict(color=ACC4, width=1.8, dash="dash")))

    y_min = -float(np.max(y)) * 0.08
    y_max = float(np.max(y)) * 1.12
    shapes = [
        # vertical at T (asymptote zero crossing)
        dict(type="line", x0=T, x1=T, y0=y_min, y1=y_max * 0.30,
             xref="x", yref="y", line=dict(color=ACC3, width=1.5, dash="dot")),
        # slope triangle: horizontal leg
        dict(type="line", x0=t_s1, x1=t_s2, y0=y_s1, y1=y_s1,
             xref="x", yref="y", line=dict(color=YELL, width=1.2, dash="dot")),
        # slope triangle: vertical leg
        dict(type="line", x0=t_s2, x1=t_s2, y0=y_s1, y1=y_s2,
             xref="x", yref="y", line=dict(color=YELL, width=1.2, dash="dot")),
    ]
    annots = [
        # T label – to the right of the vertical orange line, mid-height
        dict(x=T * 1.04, y=y_max * 0.20, xref="x", yref="y",
             text=f"T = {T:.1f} s", font=dict(color=ACC3, size=11),
             showarrow=False, xanchor="left", yanchor="middle"),
        # Δt – below the horizontal triangle leg
        dict(x=(t_s1 + t_s2) / 2, y=y_s1, xref="x", yref="y",
             text=f"Δt = {t_s2-t_s1:.1f} s", font=dict(color=YELL, size=10),
             showarrow=False, xanchor="center", yanchor="top"),
        # Δy – to the right of the vertical triangle leg, centred
        dict(x=t_s2 * 1.01, y=(y_s1 + y_s2) / 2, xref="x", yref="y",
             text=f"Δy = {y_s2-y_s1:.2f}", font=dict(color=YELL, size=10),
             showarrow=False, xanchor="left", yanchor="middle"),
        # K_I label – upper-left, well clear of the rising curves
        dict(x=t_end * 0.05, y=y_max * 0.92, xref="x", yref="y",
             text=f"K_I = Δy/Δt = {K_I:.3f}", font=dict(color=ACC4, size=11),
             showarrow=False, xanchor="left", yanchor="top"),
    ]
    fig.update_layout(
        **_BASE,
        title=dict(text="IT1 – Step Response with Identification Markers", font=dict(size=15)),
        xaxis=dict(title="Time [s]", **_AX),
        yaxis=dict(title="Output y(t)", range=[y_min, y_max], **_AX),
        shapes=shapes, annotations=annots,
        legend=_LEG,
        height=420,
    )
    return fig


# ─── PLOT: PT2 non-oscillating ────────────────────────────────────────────────
def make_pt2_plot(K: float, T1: float, T2: float) -> go.Figure:
    t, y, T1e, T2e = compute_pt2(K, T1, T2)
    t_end = float(t[-1])

    # inflection point
    if abs(T1e - T2e) < 0.05:
        t_w = T1e
        y_w = K * (1.0 - (1.0 + 1.0) * np.exp(-1.0))
        slope_w = K / (T1e * np.e)
    else:
        t_w = T1e * T2e / (T1e - T2e) * np.log(T1e / T2e)
        y_w = float(K * (1.0 - (T1e * np.exp(-t_w / T1e) - T2e * np.exp(-t_w / T2e)) / (T1e - T2e)))
        slope_w = float(K * (np.exp(-t_w / T1e) - np.exp(-t_w / T2e)) / (T1e - T2e))

    # tangent: y = slope_w * (t - t_w) + y_w
    # x-intercept (Tu): t when y=0 → t = t_w - y_w/slope_w
    Tu = t_w - y_w / slope_w
    # tangent reaches K at: t = t_w + (K - y_w)/slope_w
    t_K = t_w + (K - y_w) / slope_w
    Tg = t_K - Tu

    t_tang = np.array([max(0.0, Tu - 0.5), t_K + 0.5])
    y_tang = slope_w * (t_tang - t_w) + y_w

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Step response",
                             line=dict(color=ACC1, width=2.5)))
    fig.add_trace(go.Scatter(x=t_tang, y=y_tang, mode="lines",
                             name="Inflection-point tangent",
                             line=dict(color=ACC4, width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(x=[t_w], y=[y_w], mode="markers",
                             name="Inflection point",
                             marker=dict(color=ACC3, size=10, symbol="circle")))

    y_lo = -K * 0.16
    y_hi = K * 1.20
    brk_y = y_lo * 0.60   # y-position of the Tu / Tg brackets
    shapes = [
        dict(type="line", x0=0, x1=t_end, y0=K, y1=K, xref="x", yref="y",
             line=dict(color=YELL, width=1.5, dash="dash")),
        dict(type="line", x0=Tu, x1=Tu, y0=y_lo, y1=0,
             xref="x", yref="y", line=dict(color=ACC3, width=1.2, dash="dot")),
        dict(type="line", x0=t_K, x1=t_K, y0=y_lo, y1=K,
             xref="x", yref="y", line=dict(color=ACC3, width=1.2, dash="dot")),
        # Tu bracket
        dict(type="line", x0=0, x1=Tu, y0=brk_y, y1=brk_y,
             xref="x", yref="y", line=dict(color=ACC3, width=1.5)),
        # Tg bracket
        dict(type="line", x0=Tu, x1=t_K, y0=brk_y, y1=brk_y,
             xref="x", yref="y", line=dict(color=YELL, width=1.5)),
    ]
    annots = [
        # K label – left side, above the dashed K-line
        dict(x=t_end * 0.03, y=K * 1.02, xref="x", yref="y",
             text=f"K = {K:.2f}", font=dict(color=YELL, size=11),
             showarrow=False, xanchor="left", yanchor="bottom"),
        # Tu bracket label – centred in bracket, above bracket line
        dict(x=Tu / 2, y=brk_y, xref="x", yref="y",
             text=f"Tu = {Tu:.2f} s", font=dict(color=ACC3, size=11),
             showarrow=False, xanchor="center", yanchor="top"),
        # Tg bracket label – centred in bracket, above bracket line
        dict(x=(Tu + t_K) / 2, y=brk_y, xref="x", yref="y",
             text=f"Tg = {Tg:.2f} s", font=dict(color=YELL, size=11),
             showarrow=False, xanchor="center", yanchor="top"),
        # inflection point label – arrow pointing to the marker
        dict(x=t_w, y=y_w, xref="x", yref="y",
             text="inflection", font=dict(color=ACC3, size=10),
             showarrow=True, arrowhead=2, arrowcolor=ACC3, arrowwidth=1.2,
             ax=30, ay=-28, xanchor="left"),
    ]
    fig.update_layout(
        **_BASE,
        title=dict(text="PT2 – Step Response with Tangent Method", font=dict(size=15)),
        xaxis=dict(title="Time [s]", **_AX),
        yaxis=dict(title="Output y(t)", range=[y_lo, y_hi], **_AX),
        shapes=shapes, annotations=annots,
        legend=_LEG,
        height=420,
    )
    return fig, Tu, Tg


# ─── PLOT: PT2osc ─────────────────────────────────────────────────────────────
def make_pt2osc_plot(K: float, f_n: float, xi: float) -> go.Figure:
    t, y, wn, wd, T_d = compute_pt2osc(K, f_n, xi)
    t_end = float(t[-1])

    # first peak
    t_peak = np.pi / wd
    Mp_ratio = np.exp(-np.pi * xi / np.sqrt(1.0 - xi**2))
    y_peak = K * (1.0 + Mp_ratio)
    Mp_pct = Mp_ratio * 100.0

    # second peak (one full damped period later)
    t_peak2  = t_peak + T_d
    Mp2_ratio = np.exp(-3.0 * np.pi * xi / np.sqrt(1.0 - xi**2))
    y_peak2  = K * (1.0 + Mp2_ratio)

    # decay envelope
    env_upper = K * (1.0 + np.exp(-xi * wn * t) / np.sqrt(1.0 - xi**2))
    env_lower = K * (1.0 - np.exp(-xi * wn * t) / np.sqrt(1.0 - xi**2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=env_upper, mode="lines",
                             name="Decay envelope", line=dict(color=ACC3, width=1.0, dash="dot"),
                             showlegend=True))
    fig.add_trace(go.Scatter(x=t, y=env_lower, mode="lines",
                             line=dict(color=ACC3, width=1.0, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Step response",
                             line=dict(color=ACC1, width=2.5)))
    fig.add_trace(go.Scatter(x=[t_peak, t_peak2], y=[y_peak, y_peak2], mode="markers",
                             name=f"Peaks (1st: {Mp_pct:.1f}% overshoot)",
                             marker=dict(color=YELL, size=11, symbol="diamond")))

    y_lo = K * (0.0 - Mp_ratio * 0.55)
    y_hi = y_peak * 1.10
    brk_y = y_lo * 0.65   # y-position of the Td bracket
    shapes = [
        dict(type="line", x0=0, x1=t_end, y0=K, y1=K, xref="x", yref="y",
             line=dict(color=YELL, width=1.5, dash="dash")),
        # vertical at first peak – from x-axis to peak
        dict(type="line", x0=t_peak, x1=t_peak, y0=0, y1=y_peak, xref="x", yref="y",
             line=dict(color=YELL, width=1.2, dash="dot")),
        # vertical at second peak – same style
        dict(type="line", x0=t_peak2, x1=t_peak2, y0=0, y1=y_peak2, xref="x", yref="y",
             line=dict(color=YELL, width=1.2, dash="dot")),
        # Td bracket
        dict(type="line", x0=t_peak, x1=t_peak2, y0=brk_y, y1=brk_y,
             xref="x", yref="y", line=dict(color=ACC3, width=1.5)),
    ]
    annots = [
        # K label – left side, above the dashed K-line
        dict(x=t_end * 0.03, y=K * 1.02, xref="x", yref="y",
             text=f"K = {K:.2f}", font=dict(color=YELL, size=11),
             showarrow=False, xanchor="left", yanchor="bottom"),
        # Mp label – arrow pointing to first peak
        dict(x=t_peak, y=y_peak, xref="x", yref="y",
             text=f"Mp = {Mp_pct:.1f}%", font=dict(color=YELL, size=11),
             showarrow=True, arrowhead=2, arrowcolor=YELL, arrowwidth=1.2,
             ax=35, ay=-25, xanchor="left"),
        # Td bracket label – centred between the two peak verticals
        dict(x=(t_peak + t_peak2) / 2, y=brk_y, xref="x", yref="y",
             text=f"Td = {T_d:.3f} s", font=dict(color=ACC3, size=11),
             showarrow=False, xanchor="center", yanchor="top"),
    ]
    fig.update_layout(
        **_BASE,
        title=dict(text="PT2osc – Step Response with Identification Markers", font=dict(size=15)),
        xaxis=dict(title="Time [s]", **_AX),
        yaxis=dict(title="Output y(t)", range=[y_lo, y_hi], **_AX),
        shapes=shapes, annotations=annots,
        legend=_LEG,
        height=420,
    )
    return fig, t_peak, Mp_pct, T_d


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ System Parameters")

        sys_choice = st.selectbox(
            "System Type",
            SYS_TYPES,
            index=SYS_TYPES.index(str(safe_get("sys_type"))),
        )
        st.session_state["sys_type"] = sys_choice

        st.divider()

        if sys_choice == "PT1 – 1st Order Lag":
            st.subheader("📈 PT1 Parameters")
            v = st.slider("Gain K", PT1_K_MIN, PT1_K_MAX,
                          float(safe_get("pt1_K")), PT1_K_STEP)
            st.session_state["pt1_K"] = v
            v = st.slider("Time Constant T [s]", PT1_T_MIN, PT1_T_MAX,
                          float(safe_get("pt1_T")), PT1_T_STEP)
            st.session_state["pt1_T"] = v

        elif sys_choice == "IT1 – Integrator + Lag":
            st.subheader("📈 IT1 Parameters")
            v = st.slider("Integration Gain K_I", IT1_KI_MIN, IT1_KI_MAX,
                          float(safe_get("it1_KI")), IT1_KI_STEP)
            st.session_state["it1_KI"] = v
            v = st.slider("Lag Time Constant T [s]", IT1_T_MIN, IT1_T_MAX,
                          float(safe_get("it1_T")), IT1_T_STEP)
            st.session_state["it1_T"] = v

        elif sys_choice == "PT2 – 2nd Order Non-Oscillating":
            st.subheader("📈 PT2 Parameters")
            v = st.slider("Gain K", PT2_K_MIN, PT2_K_MAX,
                          float(safe_get("pt2_K")), PT2_K_STEP)
            st.session_state["pt2_K"] = v
            v = st.slider("Dominant Time Constant T1 [s]", PT2_T1_MIN, PT2_T1_MAX,
                          float(safe_get("pt2_T1")), PT2_T1_STEP)
            st.session_state["pt2_T1"] = v
            v = st.slider("Second Time Constant T2 [s]", PT2_T2_MIN, PT2_T2_MAX,
                          float(safe_get("pt2_T2")), PT2_T2_STEP)
            st.session_state["pt2_T2"] = v

        else:  # PT2osc
            st.subheader("📈 PT2osc Parameters")
            v = st.slider("Gain K", P2O_K_MIN, P2O_K_MAX,
                          float(safe_get("p2o_K")), P2O_K_STEP)
            st.session_state["p2o_K"] = v
            v = st.slider("Natural Frequency fn [Hz]", P2O_FN_MIN, P2O_FN_MAX,
                          float(safe_get("p2o_fn")), P2O_FN_STEP)
            st.session_state["p2o_fn"] = v
            v = st.slider("Damping Ratio ξ", P2O_XI_MIN, P2O_XI_MAX,
                          float(safe_get("p2o_xi")), P2O_XI_STEP)
            st.session_state["p2o_xi"] = v

        st.divider()
        st.subheader("💾 Save / Load")
        if st.button("💾 Save JSON", use_container_width=True):
            data = {k: safe_get(k) for k in DEFAULTS}
            st.download_button("⬇️ Download",
                               data=json.dumps(data, indent=2),
                               file_name="sysid_params.json",
                               mime="application/json",
                               use_container_width=True)
        st.markdown("**📂 Load JSON**")
        upload = st.file_uploader("Load JSON", type="json",
                                  label_visibility="collapsed")
        if upload is not None:
            _on_upload(upload)


def _on_upload(upload) -> None:
    try:
        raw = json.loads(upload.read().decode())
        for k, v in raw.items():
            if k in DEFAULTS:
                st.session_state[k] = type(DEFAULTS[k])(v)
        st.success("Parameters loaded.")
    except Exception as e:
        st.error(f"Invalid file: {e}")


# ─── TAB: PT1 ─────────────────────────────────────────────────────────────────
def tab_pt1() -> None:
    K = float(safe_get("pt1_K"))
    T = float(safe_get("pt1_T"))
    t, y = compute_pt1(K, T)

    t90 = float(t[np.argmax(y >= 0.9 * K)]) if np.any(y >= 0.9 * K) else float("nan")

    st.info(
        f"**PT1 System** | Gain K = {K:.2f}, Time Constant T = {T:.1f} s\n\n"
        f"**How to identify from a step response:**\n"
        f"1. Read the final value → that is **K = {K:.2f}**.\n"
        f"2. Find when the output reaches **63.2 % of K** → that time is **T = {T:.1f} s**.\n"
        f"3. Alternatively, draw the tangent at t = 0 — it intersects the final value at t = T."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Gain K",            f"{K:.2f}")
    c2.metric("Time Constant T",   f"{T:.1f} s")
    c3.metric("90 % Rise Time",    f"{t90:.1f} s" if not np.isnan(t90) else "—")

    st.plotly_chart(make_pt1_plot(K, T), use_container_width=True)

    with st.expander("📐 Identification Method – PT1", expanded=False):
        st.markdown("""
**Transfer function:**  G(s) = K / (1 + T·s)

**Step response:**  y(t) = K · (1 − e^(−t/T))

**Step-by-step identification from a measured curve:**

| Step | What to do | What you get |
|---|---|---|
| 1 | Read the final (steady-state) value | K |
| 2 | Find the time where y = 63.2% · K | T |
| 3 | Check: y(3T) ≈ 95%, y(5T) ≈ 99.3% | Verification |
| 4 | Draw tangent at t = 0 (slope = K/T) — it hits the final value at t = T | Cross-check T |

**Orange markers** on the plot show the 63.2% level and the corresponding time T.
**Purple dashed line** is the tangent at t = 0.
""")


# ─── TAB: IT1 ─────────────────────────────────────────────────────────────────
def tab_it1() -> None:
    K_I = float(safe_get("it1_KI"))
    T   = float(safe_get("it1_T"))
    t, y = compute_it1(K_I, T)
    t_end = float(t[-1])

    # slope of late portion (asymptote verification)
    idx = int(len(t) * 0.7)
    slope_measured = (y[-1] - y[idx]) / (t[-1] - t[idx])

    st.info(
        f"**IT1 System** | Integration Gain K_I = {K_I:.3f}, Lag T = {T:.1f} s\n\n"
        f"**How to identify from a step response:**\n"
        f"1. At large t, the output grows **linearly** — measure the slope → **K_I = {K_I:.3f}**.\n"
        f"2. Extend this line back to where it crosses y = 0 — that time is **T = {T:.1f} s**.\n"
        f"3. The ramp starts later than t = 0 by exactly T due to the lag."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Integration Gain K_I", f"{K_I:.3f}")
    c2.metric("Lag Time Constant T",  f"{T:.1f} s")
    c3.metric("Slope at t → ∞",       f"{slope_measured:.4f}")

    st.plotly_chart(make_it1_plot(K_I, T), use_container_width=True)

    with st.expander("📐 Identification Method – IT1", expanded=False):
        st.markdown("""
**Transfer function:**  G(s) = K_I / (s · (1 + T·s))

**Step response:**  y(t) = K_I · [ t − T · (1 − e^(−t/T)) ]

For large t:  y(t) ≈ K_I · (t − T)  ← a straight line (the asymptote)

**Step-by-step identification from a measured curve:**

| Step | What to do | What you get |
|---|---|---|
| 1 | Let the response run long enough to become visibly linear | Confirms IT1 type |
| 2 | Measure the slope of the linear portion | K_I = Δy / Δt |
| 3 | Extend the line backwards — find where it crosses y = 0 | T (lag) |

**Purple dashed line** is the asymptote. The slope triangle (yellow) shows how to measure K_I graphically.
The **orange vertical line** marks T, where the asymptote crosses zero.
""")


# ─── TAB: PT2 ─────────────────────────────────────────────────────────────────
def tab_pt2() -> None:
    K   = float(safe_get("pt2_K"))
    T1r = float(safe_get("pt2_T1"))
    T2r = float(safe_get("pt2_T2"))
    _, _, T1, T2 = compute_pt2(K, T1r, T2r)   # T1 >= T2 enforced inside

    fig, Tu, Tg = make_pt2_plot(K, T1r, T2r)

    st.info(
        f"**PT2 System** | K = {K:.2f}, T1 = {T1:.1f} s, T2 = {T2:.1f} s\n\n"
        f"**How to identify from a step response:**\n"
        f"1. Draw the **tangent at the inflection point** (steepest slope of the curve).\n"
        f"2. Read off **Tu = {Tu:.2f} s** (apparent delay — where the tangent crosses y = 0).\n"
        f"3. Read off **Tg = {Tg:.2f} s** (apparent rise — where the tangent crosses y = K).\n"
        f"4. From Tu and Tg, estimate T1 + T2 ≈ Tg and T1 · T2 ≈ Tu · Tg."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gain K",         f"{K:.2f}")
    c2.metric("T1",             f"{T1:.1f} s")
    c3.metric("Tu (apparent delay)",  f"{Tu:.2f} s")
    c4.metric("Tg (apparent rise)",   f"{Tg:.2f} s")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Identification Method – PT2 (Inflection Tangent)", expanded=False):
        st.markdown("""
**Transfer function:**  G(s) = K / ((1 + T1·s) · (1 + T2·s))

**Step response:**  y(t) = K · [1 − (T1·e^(−t/T1) − T2·e^(−t/T2)) / (T1 − T2)]

The curve has an **S-shape** with an inflection point — unlike a PT1 which starts immediately.

**Step-by-step identification:**

| Step | What to do | What you get |
|---|---|---|
| 1 | Find the steepest point (inflection) on the step response | Inflection time t_w |
| 2 | Draw a tangent through the inflection point | The Wendetangente |
| 3 | Read Tu: where the tangent crosses y = 0 | Apparent delay |
| 4 | Read Tg: where the tangent crosses y = K (final value) | Apparent rise time |
| 5 | Estimate: T1 + T2 ≈ Tg | Sum of time constants |
| 6 | Estimate: T1 · T2 ≈ Tu · Tg | Product (→ T2 if T1 known) |

**Rule of thumb:** If Tu/Tg < 0.1 the system behaves almost like a PT1 with T ≈ T1.
The larger Tu/Tg, the more the second pole is visible.

**Orange brackets** show Tu and Tg on the time axis. The **orange dot** is the inflection point.
""")


# ─── TAB: PT2osc ──────────────────────────────────────────────────────────────
def tab_pt2osc() -> None:
    K   = float(safe_get("p2o_K"))
    f_n = float(safe_get("p2o_fn"))
    xi  = float(safe_get("p2o_xi"))

    _, _, wn, wd, T_d = compute_pt2osc(K, f_n, xi)
    fig, t_peak, Mp_pct, T_d = make_pt2osc_plot(K, f_n, xi)

    # back-calculation from observable quantities
    xi_from_Mp  = -np.log(Mp_pct / 100.0) / np.sqrt(np.pi**2 + np.log(Mp_pct / 100.0)**2)
    f_d = 1.0 / T_d
    f_n_from_Td = f_d / np.sqrt(1.0 - xi**2)

    st.info(
        f"**PT2osc System** | K = {K:.2f}, fn = {f_n:.3f} Hz, ξ = {xi:.3f}\n\n"
        f"**How to identify from a step response:**\n"
        f"1. Measure the **first overshoot** → Mp = {Mp_pct:.1f}% → gives ξ = {xi_from_Mp:.3f}.\n"
        f"2. Measure the **oscillation period** Td = {T_d:.3f} s → damped frequency fd = {f_d:.3f} Hz.\n"
        f"3. Recover fn = fd / √(1 − ξ²) = {f_n_from_Td:.3f} Hz."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gain K",             f"{K:.2f}")
    c2.metric("Overshoot Mp",       f"{Mp_pct:.1f} %")
    c3.metric("Damped Period Td",   f"{T_d:.3f} s")
    c4.metric("Damping Ratio ξ",    f"{xi:.3f}")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Identification Method – PT2osc", expanded=False):
        st.markdown("""
**Transfer function:**  G(s) = K · ωn² / (s² + 2ξωn·s + ωn²),  with ξ < 1

**Step response:** oscillates with damped frequency ωd = ωn · √(1 − ξ²)

**Step-by-step identification:**

| Step | What to do | Formula | What you get |
|---|---|---|---|
| 1 | Read the final value | — | K |
| 2 | Measure first peak height y_p and final value K | Mp = (y_p − K) / K | Overshoot Mp |
| 3 | Compute ξ from Mp | ξ = −ln(Mp) / √(π² + ln²(Mp)) | Damping ratio ξ |
| 4 | Measure the time between consecutive peaks | Td | Damped period |
| 5 | Compute ωd = 2π / Td | — | Damped natural frequency |
| 6 | Compute ωn = ωd / √(1 − ξ²) | fn = ωn / (2π) | Natural frequency fn |

**Yellow diamond** marks the first peak. **Orange dotted lines** are the decay envelopes.
**Orange bracket** at the bottom shows one full damped period Td.

**Rule:** The decay envelope is K · (1 ± e^(−ξωn·t) / √(1−ξ²)).
When the step response just touches the envelopes, the oscillation is at its extremes.
""")


# ─── TAB: EXPLANATION ─────────────────────────────────────────────────────────
def tab_explanation() -> None:
    st.header("📖 System Identification – Theory")

    st.markdown("""
**System identification** is the process of building a mathematical model of a dynamic system
from its measured input-output behaviour. The most common approach in process control
is to analyse the **step response** — apply a step change to the input and record how the output evolves.

Each system type has a characteristic **shape** that reveals its parameters.

---
""")

    st.subheader("📈 PT1 – First Order Lag")
    st.markdown("**Transfer function:** G(s) = K / (1 + T·s)")
    st.latex(r"y(t) = K \cdot \left(1 - e^{-t/T}\right)")
    st.markdown("""
- Output rises **exponentially** without overshoot.
- **K** sets the final value (DC gain).
- **T** sets the speed — larger T means slower response.
- After 5T the output is at 99.3% of K → system is considered settled.
- **Identification:** final value = K, time to 63.2% = T.
""")

    st.divider()
    st.subheader("📈 IT1 – Integrator + First Order Lag")
    st.markdown("**Transfer function:** G(s) = K_I / (s · (1 + T·s))")
    st.latex(r"y(t) = K_I \cdot \left[t - T \cdot \left(1 - e^{-t/T}\right)\right]")
    st.markdown("""
- Output **ramps upward indefinitely** — never reaches a steady state for a constant input.
- This is typical of **integrating processes** (e.g. a tank being filled, a motor position).
- For large t the response is linear with slope K_I — the lag T only shows up as a short initial curve.
- **Identification:** measure the slope of the linear portion (= K_I), extend the line back to find T.
""")

    st.divider()
    st.subheader("📈 PT2 – 2nd Order Non-Oscillating (Overdamped)")
    st.markdown("**Transfer function:** G(s) = K / ((1 + T1·s) · (1 + T2·s))")
    st.latex(r"y(t) = K \cdot \left[1 - \frac{T_1\,e^{-t/T_1} - T_2\,e^{-t/T_2}}{T_1 - T_2}\right]")
    st.markdown("""
- Output rises in an **S-shape** — it starts slow, accelerates, then flattens out.
- The S-shape appears because two time constants are in series — one limits the rise, the other the approach to the final value.
- **Identification:** draw the tangent at the inflection point (steepest slope).
  - Tu (apparent delay) = where the tangent crosses y = 0
  - Tg (apparent rise) = where the tangent crosses y = K
  - Approximate: T1 + T2 ≈ Tg and T1 · T2 ≈ Tu · Tg
""")

    st.divider()
    st.subheader("📈 PT2osc – 2nd Order Oscillating (Underdamped)")
    st.markdown("**Transfer function:** G(s) = K · ωn² / (s² + 2ξωn·s + ωn²),  ξ < 1")
    st.latex(r"y(t) = K \cdot \left[1 - e^{-\xi\omega_n t}\left(\cos\omega_d t + \frac{\xi}{\sqrt{1-\xi^2}}\sin\omega_d t\right)\right]")
    st.markdown(r"""
where ωd = ωn · √(1 − ξ²) is the **damped natural frequency**.

- Output **overshoots** and oscillates before settling — characteristic of lightly damped mechanical or electrical resonances.
- **ξ (damping ratio)** controls how quickly oscillations decay: small ξ → many oscillations, ξ → 1 → no overshoot.
- **fn (natural frequency)** controls how fast the system oscillates.
- **Identification:**
  - First overshoot Mp → ξ via  ξ = −ln(Mp) / √(π² + ln²(Mp))
  - Oscillation period Td → ωd = 2π/Td → ωn = ωd / √(1−ξ²)
""")

    st.divider()
    st.markdown("""
### Summary Table

| | PT1 | IT1 | PT2 | PT2osc |
|---|---|---|---|---|
| Poles | 1 real | 1 real + integrator | 2 real | 2 complex |
| Steady state? | Yes | No (ramps) | Yes | Yes |
| Overshoot? | No | No | No | Yes |
| S-shaped? | No | — | Yes | — |
| Key markers | 63.2% → T | slope → K_I | Tu, Tg | Mp → ξ, Td → fn |
""")


# ─── TAB: EXERCISES ───────────────────────────────────────────────────────────
def tab_exercises() -> None:
    st.header("✏️ Exercises")
    st.markdown("Use the **sidebar** to adjust parameters. Observe the step response and read off the identification markers.")

    with st.expander("🔵 Exercise 1 – PT1: Read K and T", expanded=True):
        st.markdown("""
**Setup:** Set K = 3.0, T = 8.0 s on the PT1 tab.

**Tasks:**
1. What is the final value of the step response?
2. At which time does the output reach 63.2% of the final value?
3. Draw (mentally) the tangent at t = 0. Where does it intersect the final value?
""")
        with st.expander("💡 Solution", expanded=False):
            st.markdown("""
1. Final value = **K = 3.0** — read directly from the yellow dashed line.
2. 63.2% of 3.0 = 1.896 → this is reached at **t = T = 8.0 s** — shown by the orange marker.
3. The tangent has slope K/T = 3.0/8.0 = 0.375. It crosses y = 3.0 at t = T = **8.0 s**.

All three methods give the same T — they are just different ways to read the same parameter.
""")

    with st.expander("🔵 Exercise 2 – PT1: Effect of T on Speed", expanded=False):
        st.markdown("""
**Setup:** K = 2.0, vary T between 2 s and 20 s.

**Tasks:**
1. How does the step response shape change with T?
2. Double T from 5 s to 10 s — how does the 90% rise time change?
3. Does changing T affect the final value?
""")
        with st.expander("💡 Solution", expanded=False):
            st.markdown("""
1. Larger T → slower, more gradual rise. The shape stays exponential, just stretched in time.
2. 90% rise time ≈ 2.3·T. Doubling T from 5→10 s doubles the 90% rise time from ~11.5 s to ~23 s.
3. **No.** The final value is always K, regardless of T. T only controls how fast the output gets there.

**Key insight:** K and T are **independent** — K scales the output vertically, T scales it horizontally.
""")

    with st.expander("🟣 Exercise 3 – IT1: Identify K_I and T", expanded=False):
        st.markdown("""
**Setup:** Set K_I = 0.5, T = 5.0 s on the IT1 tab.

**Tasks:**
1. Let the simulation run. Does the output settle to a fixed value?
2. At large t, the output is linear. Measure Δy over a time interval Δt and compute K_I.
3. Extend the linear portion backwards to y = 0. At which time does it cross?
""")
        with st.expander("💡 Solution", expanded=False):
            st.markdown("""
1. **No** — the output keeps rising indefinitely. An IT1 has an integrator: constant input → output ramps forever.
2. At large t: slope = K_I = **0.5** [units/s]. For example, from t = 20 to t = 40 s, y rises by 10 → slope = 10/20 = 0.5 ✓
3. The asymptote y = K_I·(t − T) = 0.5·(t − 5) crosses zero at **t = T = 5.0 s** — the orange line.

**Key insight:** The IT1 looks like a ramp with a short initial curve. The lag T is invisible once the system settles into its ramp — you must extend the line back to find it.
""")

    with st.expander("🟢 Exercise 4 – PT2: Tangent Method", expanded=False):
        st.markdown("""
**Setup:** K = 2.0, T1 = 10.0 s, T2 = 2.0 s on the PT2 tab.

**Tasks:**
1. Does the step response start immediately (like PT1) or with a delay?
2. Read off Tu (apparent delay) and Tg (apparent rise time) from the plot.
3. Estimate T1 + T2 from Tg and T1 · T2 from Tu · Tg. Compare to the actual values.
""")
        with st.expander("💡 Solution", expanded=False):
            st.markdown("""
1. The response has an **S-shape** — it starts with zero slope (unlike PT1), rises, then flattens. There is apparent dead-time even though none was specified.

2. For T1 = 10, T2 = 2:
   - Inflection at t_w = T1·T2/(T1−T2) · ln(T1/T2) = 20/8 · ln(5) ≈ 2.5 · 1.609 ≈ **4.02 s**
   - Approximate Tu ≈ 1.5 s, Tg ≈ 10.5 s (read from plot markers)

3. T1 + T2 = 12 ≈ Tg ✓ | T1 · T2 = 20 ≈ Tu · Tg ≈ 1.5 · 10.5 = 15.8 (rough approximation)

**Note:** The tangent method gives approximate values. The closer T1 and T2 are, the less accurate the approximation becomes.
""")

    with st.expander("🟢 Exercise 5 – PT2osc: Identify ξ and fn from Overshoot", expanded=False):
        st.markdown("""
**Setup:** K = 1.0, fn = 0.5 Hz, ξ = 0.2 on the PT2osc tab.

**Tasks:**
1. Read the first overshoot Mp from the plot (in %).
2. Use the formula ξ = −ln(Mp/100) / √(π² + ln²(Mp/100)) to compute ξ.
3. Read the damped period Td and compute fn.
""")
        with st.expander("💡 Solution", expanded=False):
            st.markdown(r"""
For ξ = 0.2:

1. Mp = exp(−π · 0.2 / √(1 − 0.04)) × 100 = exp(−0.644) × 100 ≈ **52.5%**

2. ξ from Mp: ln(0.525) = −0.644 → ξ = 0.644 / √(π² + 0.415) = 0.644 / √(10.27) ≈ **0.201** ✓

3. ωd = ωn · √(1 − ξ²) = 2π · 0.5 · √(0.96) ≈ 3.084 rad/s
   Td = 2π / ωd ≈ **2.038 s**
   From Td: fn = (1/Td) / √(1 − ξ²) = 0.4907 / 0.98 ≈ **0.500 Hz** ✓

**Key insight:** Just two measurements from the step response (Mp and Td) are enough to fully identify a 2nd order oscillating system.
""")

    with st.expander("🟢 Exercise 6 – PT2osc: Effect of Damping Ratio ξ", expanded=False):
        st.markdown("""
**Setup:** K = 1.0, fn = 0.5 Hz. Vary ξ from 0.1 to 0.9.

**Tasks:**
1. At ξ = 0.1 — how many oscillations are visible before settling?
2. At which ξ does the overshoot practically disappear?
3. Compare ξ = 0.7 and ξ = 0.9 — which settles faster?
""")
        with st.expander("💡 Solution", expanded=False):
            st.markdown("""
1. At ξ = 0.1: many oscillations — the system is very lightly damped. The decay envelope is wide and decays slowly.

2. Overshoot becomes negligible around **ξ ≈ 0.7–0.8**. At ξ = 0.707 (Butterworth) the response has no overshoot and the fastest settling without oscillation.

3. ξ = 0.7 settles faster than ξ = 0.9. Counter-intuitive: **increasing ξ beyond ~0.7 slows the response** because the system becomes overdamped and approaches the final value sluggishly from below.

**Design rule:** ξ ≈ 0.5–0.7 is typically the best trade-off between speed and overshoot in control system design.
""")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    render_sidebar()

    st.title("🔍 System Identification from Step Responses")

    st.markdown("""
Select a **System Type** in the sidebar — only the relevant parameters will appear.
Switch between tabs to view the step response, read theory, or work through exercises.

| System | Shape | Key parameters |
|---|---|---|
| **PT1** | Exponential rise, no overshoot | Gain K, time constant T |
| **IT1** | Linear ramp (integrating) | Integration gain K_I, lag T |
| **PT2** | S-shaped rise, no overshoot | Gain K, apparent delay Tu, rise time Tg |
| **PT2osc** | Oscillating, overshoots | Gain K, natural frequency fn, damping ξ |
""")
    st.divider()

    tabs = st.tabs(["📊 Step Response", "📖 Explanation", "✏️ Exercises"])

    with tabs[0]:
        sys_type = str(safe_get("sys_type"))
        if sys_type == "PT1 – 1st Order Lag":
            tab_pt1()
        elif sys_type == "IT1 – Integrator + Lag":
            tab_it1()
        elif sys_type == "PT2 – 2nd Order Non-Oscillating":
            tab_pt2()
        else:
            tab_pt2osc()

    with tabs[1]: tab_explanation()
    with tabs[2]: tab_exercises()


if __name__ == "__main__":
    main()
