import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import fetch_market_data
from src.qubo_builder import build_portfolio_qp, qp_to_ising_operator
from src.solvers import solve_classical, solve_qaoa
from src.metrics import calculate_portfolio_metrics

# ---------------------------------------------------------
# Page Configuration & Institutional Theme Overrides
# ---------------------------------------------------------
st.set_page_config(
    page_title="Q-Markowitz | Quantum Portfolio Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f1f5f9;
    }
    
    code, .metric-value, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #090d16;
    }
    
    /* -------------------------------------------------- */
    /* SIDEBAR REFINEMENT (OVERRIDING DEFAULT RED ACCENTS)*/
    /* -------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
        padding-top: 1.5rem;
    }

    /* Sidebar Headings & Section Title */
    .sidebar-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
    }

    /* Labels styling */
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 4px !important;
    }

    /* Multiselect Box Container */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px !important;
        min-height: 42px !important;
    }

    /* Multiselect Tags (Pills) - Eliminating Harsh Red */
    section[data-testid="stSidebar"] span[data-baseweb="tag"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 4px !important;
        padding: 3px 8px !important;
        margin: 2px !important;
    }
    
    section[data-testid="stSidebar"] span[data-baseweb="tag"] span {
        color: #93c5fd !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
    }
    
    section[data-testid="stSidebar"] span[data-baseweb="tag"] svg {
        fill: #64748b !important;
    }
    
    section[data-testid="stSidebar"] span[data-baseweb="tag"] svg:hover {
        fill: #f87171 !important;
    }

    /* Sliders Overrides (Rail, Thumb, and Value Numbers) */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] {
        margin-top: 4px;
        margin-bottom: 12px;
    }

    /* Slider thumb knob */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #3b82f6 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.4) !important;
        width: 14px !important;
        height: 14px !important;
    }

    /* Slider numeric text value */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        color: #60a5fa !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    /* Dropdown Selectbox */
    section[data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #e2e8f0 !important;
        font-size: 0.82rem !important;
    }

    /* Primary Run Action Button */
    section[data-testid="stSidebar"] div.stButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.65rem 1rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
        margin-top: 10px;
    }
    
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
    }

    /* Metadata Specification Card at Sidebar Bottom */
    .sidebar-meta-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 12px 14px;
        margin-top: 1.5rem;
    }

    /* -------------------------------------------------- */
    /* MAIN CANVAS CARDS & STATUS                         */
    /* -------------------------------------------------- */
    .card-container {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    
    .metric-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.55rem;
        font-weight: 600;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    .status-badge-optimal {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        background-color: rgba(16, 185, 129, 0.08);
        border: 1px solid #10b981;
        color: #10b981;
    }

    .status-badge-suboptimal {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        background-color: rgba(245, 158, 11, 0.08);
        border: 1px solid #f59e0b;
        color: #f59e0b;
    }

    .stTable, [data-testid="stDataFrame"] {
        border-radius: 6px;
        border: 1px solid #1e293b;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: #1e293b;
        margin: 20px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Configuration Panel</div>", unsafe_allow_html=True)
    
    universe_options = [
        "BBCA.JK", "TLKM.JK", "ASII.JK", "BMRI.JK", "BBNI.JK", "UNVR.JK", 
        "AAPL", "MSFT", "GOOGL", "NVDA"
    ]
    selected_tickers = st.multiselect(
        "Asset Universe (Yahoo Finance Tickers)",
        options=universe_options,
        default=["BBCA.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"]
    )
    
    max_budget = max(1, len(selected_tickers) - 1) if len(selected_tickers) > 1 else 1
    budget = st.slider(
        "Target Portfolio Size (Budget)",
        min_value=1,
        max_value=max_budget,
        value=min(2, max_budget),
        step=1
    )
    
    risk_factor = st.slider(
        "Risk Aversion Coefficient (q)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="0.0 enforces pure return maximization; 1.0 strictly minimizes portfolio covariance."
    )
    
    qaoa_reps = st.selectbox(
        "QAOA Circuit Depth (p-layers)",
        options=[1, 2, 3],
        index=0
    )
    
    execute_button = st.button("RUN OPTIMIZATION ENGINE")
    
    st.markdown(
        """
        <div class="sidebar-meta-card">
            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; margin-bottom:6px;">System Pipeline</div>
            <div style="font-size:0.75rem; color:#cbd5e1; line-height:1.6; font-family:'JetBrains Mono', monospace;">
                • Engine: Qiskit Statevector<br>
                • Solver: COBYLA (100 iters)<br>
                • Mapping: QUBO to Ising
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Data Caching & Header
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_market_data(tickers):
    return fetch_market_data(tickers)

st.markdown("<h2 style='font-weight:700; margin-bottom:2px; letter-spacing:-0.02em;'>Q-Markowitz Optimization Platform</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:0.92rem; margin-bottom:20px;'>Hybrid Quantum-Classical (QAOA) Benchmarking against Classical Combinatorial Eigensolvers</p>", unsafe_allow_html=True)

if not execute_button:
    st.markdown(
        """
        <div class="card-container" style="border-left: 3px solid #2563eb;">
            <p style="font-weight: 600; margin-bottom: 4px; font-size: 0.9rem; color: #f8fafc;">System Initialized</p>
            <p style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 0;">
                Select asset constituents and risk-aversion parameters in the left panel, then trigger the execution pipeline to benchmark classical combinatorial diagonalization against variational quantum state evaluation.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.stop()

if len(selected_tickers) < 2:
    st.error("Invalid input: Please select at least two assets.")
    st.stop()

# ---------------------------------------------------------
# Computation Pipeline
# ---------------------------------------------------------
with st.spinner("Executing variational optimization and compiling probability spectrum..."):
    mu, sigma, tickers = load_market_data(selected_tickers)
    qp = build_portfolio_qp(tickers, mu, sigma, risk_factor=risk_factor, budget=budget)
    operator, offset, _ = qp_to_ising_operator(qp)
    
    res_classical = solve_classical(qp, tickers)
    res_qaoa = solve_qaoa(operator, qp, tickers, reps=qaoa_reps, maxiter=100)
    
    metrics_classical = calculate_portfolio_metrics(res_classical["binary_vector"], mu, sigma, tickers)
    metrics_qaoa = calculate_portfolio_metrics(res_qaoa["binary_vector"], mu, sigma, tickers)
    
    is_optimal_match = np.array_equal(res_classical["binary_vector"], res_qaoa["binary_vector"])

# ---------------------------------------------------------
# Executive Status Banner
# ---------------------------------------------------------
banner_html = f"""
<div style="display:flex; justify-content:space-between; align-items:center; background:#111827; border:1px solid #1e293b; border-radius:6px; padding:14px 18px; margin-bottom:20px;">
    <div>
        <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#94a3b8; font-weight:600;">Solution Verifier</span>
        <div style="font-size:1.02rem; font-weight:600; color:#f8fafc; margin-top:2px;">
            {'Quantum Solution Matches Classical Ground Truth Exactly' if is_optimal_match else 'Quantum Solution Settled in Sub-Optimal Local Minimum'}
        </div>
    </div>
    <div>
        <span class="{'status-badge-optimal' if is_optimal_match else 'status-badge-suboptimal'}">
            {'OPTIMAL (100% MATCH)' if is_optimal_match else f'SUB-OPTIMAL (DEPTH p={qaoa_reps})'}
        </span>
    </div>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card-container">
        <div class="metric-title">Ground Truth Energy</div>
        <div class="metric-value">{res_classical['objective_value']:.4f}</div>
        <div class="metric-sub">Classical Exact Solver</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card-container">
        <div class="metric-title">QAOA Evaluated Energy</div>
        <div class="metric-value">{res_qaoa['objective_value']:.4f}</div>
        <div class="metric-sub">Circuit Depth p={qaoa_reps}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card-container">
        <div class="metric-title">Annualized Return</div>
        <div class="metric-value">{metrics_qaoa['return_annual']*100:.2f}%</div>
        <div class="metric-sub">Volatility: {metrics_qaoa['volatility_annual']*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card-container">
        <div class="metric-title">Portfolio Sharpe Ratio</div>
        <div class="metric-value">{metrics_qaoa['sharpe_ratio']:.3f}</div>
        <div class="metric-sub">Risk-free Rate = 6.0%</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Comparative Solver Performance & Donut Allocation
# ---------------------------------------------------------
col_table, col_donut = st.columns([1.15, 0.85])

with col_table:
    st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#f8fafc; margin-bottom:10px;'>Comparative Solver Performance</p>", unsafe_allow_html=True)
    
    comp_data = {
        "Metric / Parameter": [
            "Selected Constituents",
            "Binary Vector (x)",
            "Markowitz Objective Function",
            "Expected Annual Return",
            "Annual Volatility (Risk)",
            "Sharpe Ratio (Rf=6%)",
            "Budget Constraint Feasibility"
        ],
        "Classical Exact": [
            ", ".join(res_classical["selected_tickers"]),
            np.array2string(res_classical["binary_vector"]),
            f"{res_classical['objective_value']:.5f}",
            f"{metrics_classical['return_annual']*100:.2f}%",
            f"{metrics_classical['volatility_annual']*100:.2f}%",
            f"{metrics_classical['sharpe_ratio']:.3f}",
            "Feasible"
        ],
        f"QAOA (p={qaoa_reps})": [
            ", ".join(res_qaoa["selected_tickers"]),
            np.array2string(res_qaoa["binary_vector"]),
            f"{res_qaoa['objective_value']:.5f}",
            f"{metrics_qaoa['return_annual']*100:.2f}%",
            f"{metrics_qaoa['volatility_annual']*100:.2f}%",
            f"{metrics_qaoa['sharpe_ratio']:.3f}",
            "Feasible" if res_qaoa["is_feasible"] else "Violated"
        ]
    }
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

with col_donut:
    st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#f8fafc; margin-bottom:10px;'>Target Capital Allocation (QAOA)</p>", unsafe_allow_html=True)
    
    if len(metrics_qaoa["allocations"]) > 0:
        labels = list(metrics_qaoa["allocations"].keys())
        values = list(metrics_qaoa["allocations"].values())
        palette = ['#2563eb', '#0d9488', '#4f46e5', '#0284c7', '#10b981', '#64748b']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=palette[:len(labels)], line=dict(color='#090d16', width=2)),
            showlegend=False
        )])
        
        fig_donut.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=260,
            font=dict(family="Inter", size=11, color="#94a3b8")
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
    else:
        st.caption("No valid asset allocation identified.")

# ---------------------------------------------------------
# Quantum State Probability Spectrum
# ---------------------------------------------------------
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#f8fafc; margin-bottom:4px;'>Quantum State Probability Spectrum</p>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; font-size:0.8rem; margin-bottom:14px;'>Distribution of measurement probabilities across computational basis states. The dominant eigenstate corresponds to the evaluated ground-state portfolio.</p>", unsafe_allow_html=True)

prob_dict = res_qaoa["probabilities"]
top_states = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:7]

labels_clean = []
probs = []
is_best_list = []

for idx, (bitstring, prob) in enumerate(reversed(top_states)):
    bin_arr = [int(b) for b in reversed(bitstring)]
    selected_names = [tickers[i] for i, b in enumerate(bin_arr) if b == 1]
    
    is_budget_ok = (sum(bin_arr) == budget)
    tickers_str = " + ".join(selected_names) if selected_names else "Empty Portfolio"
    
    status_tag = "" if is_budget_ok else " [Violated]"
    display_title = f"{tickers_str} ({bitstring}){status_tag}"
    
    labels_clean.append(display_title)
    probs.append(prob)
    is_best_list.append(idx == len(top_states) - 1)

bar_colors = ['#2563eb' if is_best else '#1e293b' for is_best in is_best_list]
border_colors = ['#60a5fa' if is_best else '#334155' for is_best in is_best_list]

fig_bar = go.Figure(data=[go.Bar(
    y=labels_clean,
    x=probs,
    orientation='h',
    text=[f"{p*100:.1f}%" for p in probs],
    textposition='outside',
    marker=dict(
        color=bar_colors,
        line=dict(color=border_colors, width=1)
    )
)])

fig_bar.update_layout(
    margin=dict(t=10, b=20, l=10, r=40),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=290,
    font=dict(family="JetBrains Mono", size=11, color="#cbd5e1"),
    xaxis=dict(
        title="Measurement Probability",
        showgrid=True,
        gridcolor='#1e293b',
        zeroline=False,
        tickformat='.0%',
        range=[0, max(probs) * 1.28]
    ),
    yaxis=dict(
        showgrid=False,
        linecolor='#1e293b'
    )
)

st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})