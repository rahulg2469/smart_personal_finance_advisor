import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import datetime
import sys
import os
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.auth import login_page, signup_page, logout
from dashboard.models import (
    get_user_cluster,
    get_budget_recommendations,
    get_priority_advice,
    detect_anomalies,
    predict_next_month,
    categorize_transactions,
    summarize_by_category,
    RF_LOADED
)

SUPABASE_URL = "https://lkcbsdngpbpffnxtnlys.supabase.co"
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

CATEGORIES = [
    'grocery_pos', 'grocery_net', 'entertainment', 'shopping_pos',
    'shopping_net', 'food_dining', 'gas_transport', 'health_fitness',
    'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care', 'travel'
]

CATEGORY_LABELS = {
    'grocery_pos': 'Groceries (In-store)',
    'grocery_net': 'Groceries (Online)',
    'entertainment': 'Entertainment',
    'shopping_pos': 'Shopping (In-store)',
    'shopping_net': 'Shopping (Online)',
    'food_dining': 'Food & Dining',
    'gas_transport': 'Gas & Transport',
    'health_fitness': 'Health & Fitness',
    'home': 'Home',
    'kids_pets': 'Kids & Pets',
    'misc_net': 'Misc (Online)',
    'misc_pos': 'Misc (In-store)',
    'personal_care': 'Personal Care',
    'travel': 'Travel'
}

st.set_page_config(
    page_title="Smart Finance Advisor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'user' not in st.session_state:
    st.session_state.user = None
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'demo_user' not in st.session_state:
    st.session_state.demo_user = None
if 'savings_goal' not in st.session_state:
    st.session_state.savings_goal = 500

LOGO_PATH = "dashboard/static/darklogo.png"

def get_logo_b64():
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, .stApp {
        background: #080e1a !important;
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }

    /* ── Landing typography ── */
    .page-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.8rem; font-weight: 600;
        color: #f5e6c0; letter-spacing: 0.06em;
        text-align: center; margin: 0 0 6px; line-height: 1.15;
    }
    .advisor-label {
        display: flex; align-items: center; justify-content: center;
        gap: 12px; margin-bottom: 14px;
    }
    .advisor-label .line { width:44px; height:1px; background:linear-gradient(90deg,transparent,#b8963e); }
    .advisor-label .line.r { background:linear-gradient(270deg,transparent,#b8963e); }
    .advisor-label .text {
        font-size:0.7rem; letter-spacing:0.28em; color:#b8963e;
        text-transform:uppercase; font-weight:500;
    }
    .hero-desc {
        font-size:1rem; color:#94a3b8; text-align:center;
        max-width:520px; line-height:1.75; margin:0 auto 28px;
    }

    /* ── Mode cards ── */
    .mode-card {
        background: rgba(15,23,42,0.95); border-radius:14px;
        padding:26px 22px 18px; border:1px solid rgba(51,65,85,0.8);
    }
    .mode-card.demo { border-color:rgba(59,130,246,0.45); }
    .mode-card.acct { border-color:rgba(16,185,129,0.4); }
    .mc-header { display:flex; align-items:center; gap:10px; margin-bottom:11px; }
    .mc-icon {
        width:32px; height:32px; border-radius:8px;
        display:flex; align-items:center; justify-content:center; font-size:16px;
    }
    .mc-icon.blue { background:linear-gradient(135deg,#1e3a8a,#3b82f6); }
    .mc-icon.green { background:linear-gradient(135deg,#065f46,#10b981); }
    .mc-title { font-size:1.1rem; font-weight:600; color:#f1f5f9; }
    .mc-desc { color:#94a3b8; font-size:0.86rem; line-height:1.65; margin-bottom:14px; }
    .mc-list { list-style:none; padding:0; margin:0 0 18px; }
    .mc-list li { display:flex; align-items:center; gap:8px; color:#cbd5e1; font-size:0.86rem; margin-bottom:8px; }
    .ck-b { color:#3b82f6; font-weight:700; }
    .ck-g { color:#10b981; font-weight:700; }

    /* ── Features strip ── */
    .feat-strip {
        background:rgba(15,23,42,0.7); border:1px solid rgba(51,65,85,0.55);
        border-radius:14px; padding:24px 30px;
        display:grid; grid-template-columns:1fr 1fr 1fr; gap:18px; margin-top:18px;
    }
    .feat-item { text-align:center; }
    .feat-icon {
        width:40px; height:40px; border-radius:10px; margin:0 auto 9px;
        display:flex; align-items:center; justify-content:center; font-size:18px;
    }
    .feat-name { font-weight:600; color:#e2e8f0; font-size:0.86rem; margin-bottom:4px; }
    .feat-desc { color:#64748b; font-size:0.75rem; line-height:1.5; }

    /* ── Demo page ── */
    .page-header {
        font-family:'Cormorant Garamond',serif;
        font-size:1.7rem; font-weight:600; color:#f1f5f9;
        letter-spacing:0.02em; margin:0; line-height:1;
    }
    .section-label {
        font-size:0.72rem; color:#64748b; letter-spacing:0.18em;
        text-transform:uppercase; font-weight:500; margin-bottom:9px;
    }
    .spend-banner {
        background:rgba(15,23,42,0.9); border:1px solid rgba(51,65,85,0.6);
        border-radius:12px; padding:18px 22px;
    }
    .spend-label { font-size:0.7rem; color:#64748b; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:3px; }
    .spend-amount {
        font-family:'DM Sans', sans-serif;
        font-size:2.4rem; font-weight:700; color:#f1f5f9; line-height:1;
    }
    .a-card {
        background:rgba(15,23,42,0.9); border:1px solid rgba(51,65,85,0.6);
        border-radius:12px; padding:18px 18px 14px;
    }
    /* Header-only card — no bottom border shown, just the title label */
    .section-card {
        padding: 4px 0 8px;
    }
    .a-card-title {
        font-size:0.93rem; font-weight:600; color:#f1f5f9;
        margin-bottom:13px; display:flex; align-items:center; gap:7px;
    }
    .cluster-name {
        font-family:'Cormorant Garamond',serif;
        font-size:1.5rem; font-weight:700; color:#3b82f6; margin-bottom:7px;
    }
    .cluster-desc {
        font-size:0.82rem; color:#64748b; line-height:1.6;
        background:rgba(30,41,59,0.7); border-radius:8px; padding:10px 12px;
    }
    .savings-val {
        font-family:'Cormorant Garamond',serif;
        font-size:1.85rem; font-weight:700; color:#3b82f6; margin:5px 0 16px;
    }
    .rec-row {
        background:rgba(30,41,59,0.55); border:1px solid rgba(51,65,85,0.45);
        border-radius:9px; padding:12px 16px;
        display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;
    }
    .rec-name { font-weight:600; color:#e2e8f0; font-size:0.9rem; }
    .rec-right { text-align:right; }
    .rec-amts { font-size:0.78rem; color:#94a3b8; margin-bottom:2px; }
    .rec-save { font-size:0.82rem; font-weight:600; color:#10b981; }
    .pri-row {
        display:flex; align-items:center; gap:10px;
        padding:8px 0; border-bottom:1px solid rgba(51,65,85,0.3);
    }
    .pri-rank {
        width:20px; height:20px; border-radius:5px; background:rgba(30,41,59,0.8);
        display:flex; align-items:center; justify-content:center;
        font-size:0.7rem; font-weight:600; color:#64748b; flex-shrink:0;
    }
    .pri-cat { flex:1; font-size:0.84rem; color:#cbd5e1; }
    .diff-badge { font-size:0.7rem; font-weight:600; padding:2px 9px; border-radius:20px; }
    .d-easy   { background:rgba(16,185,129,0.15); color:#10b981; }
    .d-medium { background:rgba(234,179,8,0.15);  color:#eab308; }
    .d-hard   { background:rgba(239,68,68,0.15);  color:#ef4444; }
    .anom-header { display:flex; align-items:center; gap:7px; margin-bottom:9px; }
    .anom-title  { font-size:0.93rem; font-weight:600; color:#f59e0b; }
    .anom-sub    { font-size:0.79rem; color:#64748b; margin-bottom:9px; }
    .anom-row {
        background:rgba(30,41,59,0.5); border:1px solid rgba(245,158,11,0.2);
        border-radius:8px; padding:10px 13px;
        display:flex; align-items:flex-start; gap:8px; margin-bottom:6px;
    }
    .anom-dot { width:7px; height:7px; border-radius:50%; background:#f59e0b; flex-shrink:0; margin-top:5px; }
    .anom-text { font-size:0.84rem; color:#fbbf24; line-height:1.5; }
    .no-anom {
        background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
        border-radius:8px; padding:11px 14px; color:#10b981; font-size:0.84rem;
        display:flex; align-items:center; gap:7px;
    }

    /* ── Buttons ── */
    .stButton > button {
        width:100%; border-radius:9px; border:none;
        font-weight:600; font-size:0.9rem; height:44px;
        font-family:'DM Sans',sans-serif; transition:all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
        color:white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background:linear-gradient(135deg,#1d4ed8,#1e40af) !important;
        box-shadow:0 4px 16px rgba(37,99,235,0.4); transform:translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background:linear-gradient(135deg,#059669,#047857) !important;
        color:white !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background:linear-gradient(135deg,#047857,#065f46) !important;
        box-shadow:0 4px 16px rgba(5,150,105,0.4); transform:translateY(-1px);
    }

    /* ── Back button on demo page — grey, small ── */
    [data-testid="column"]:first-child .stButton > button[kind="secondary"] {
        background: rgba(30,41,59,0.85) !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
        box-shadow: none !important;
        height: 38px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    [data-testid="column"]:first-child .stButton > button[kind="secondary"]:hover {
        background: rgba(51,65,85,0.8) !important;
        color: #cbd5e1 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── User pill buttons — grey, NOT green ── */
    .user-pills [data-testid="column"] .stButton > button {
        background: rgba(30,41,59,0.85) !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        height: 46px; font-weight: 500; font-size: 0.86rem;
    }
    .user-pills [data-testid="column"] .stButton > button:hover {
        background: rgba(37,99,235,0.15) !important;
        border-color: #3b82f6 !important;
        color: #93c5fd !important;
        transform: none !important;
    }

    /* ── Slider: make it blue ── */
    div[data-testid="stSlider"] { padding-top:4px; }
    div[data-testid="stSlider"] > div > div > div {
        background: #2563eb !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }

    /* ── Hide heading anchor link icon ── */
    h1 a, h2 a, h3 a, .page-title a,
    [data-testid="stHeadingWithActionElements"] a,
    .stMarkdown h1 a { display: none !important; }

    /* ── Hero text centered ── */
    .hero-desc {
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }

    /* ── Remove logo square background ── */
    [data-testid="stImage"] {
        background: transparent !important;
    }

    /* ── Back button on auth page: small grey pill ── */
    .back-pill > .stButton > button {
        background: rgba(30,41,59,0.8) !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        height: 36px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        width: auto !important;
        padding: 0 14px !important;
    }
    .back-pill > .stButton > button:hover {
        background: rgba(51,65,85,0.8) !important;
        color: #cbd5e1 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
def main():
    inject_css()
    if st.session_state.mode is None:
        landing_page()
    elif st.session_state.mode == 'demo':
        demo_page()
    elif st.session_state.mode == 'auth':
        if st.session_state.user:
            sidebar_nav()
            user_dashboard()
        else:
            auth_page()


# ─────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────
def landing_page():
    # Logo — pure HTML img, no Streamlit image toolbar, centered, floating
    logo_b64 = get_logo_b64()
    if logo_b64:
        st.markdown(f"""
        <div style="display:flex; justify-content:center; margin-bottom:4px;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="width:200px; height:200px; object-fit:contain;
                        animation:float 4s ease-in-out infinite;
                        filter:drop-shadow(0 0 24px rgba(212,175,55,0.2));"
                 draggable="false" />
        </div>
        <style>
        @keyframes float {{
            0%   {{ transform: translateY(0px); }}
            50%  {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
    <h1 class="page-title">Smart Finance Advisor</h1>
    <div class="advisor-label">
        <span class="line"></span>
        <span class="text">AI-Powered Personal Finance</span>
        <span class="line r"></span>
    </div>
    <p class="hero-desc" style="text-align:center;margin:0 auto 28px;max-width:520px;display:block;">
        Take control of your spending with intelligent insights.
        Track your budget, get personalized recommendations,
        and build better financial habits.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="mode-card demo">
            <div class="mc-header">
                <div class="mc-icon blue"><img src="./app/static/bar-graph.png" width="20" style="filter: brightness(0) invert(1);"></div>
                <div class="mc-title">Demo Mode</div>
            </div>
            <div class="mc-desc">Explore with sample users from our dataset. No signup needed.
            See real spending patterns and get recommendations instantly.</div>
            <ul class="mc-list">
                <li><span class="ck-b">✓</span> Browse sample spending data</li>
                <li><span class="ck-b">✓</span> View AI-powered analysis</li>
                <li><span class="ck-b">✓</span> Get budget recommendations</li>
            </ul>
        </div>
        <div style="height:10px"></div>
        """, unsafe_allow_html=True)
        if st.button("Try Demo  →", key="demo_btn", type="primary", use_container_width=True):
            st.session_state.mode = 'demo'
            st.rerun()

    with col2:
        st.markdown("""
        <div class="mode-card acct">
            <div class="mc-header">
                <div class="mc-icon green"><img src="./app/static/login.png" width="20" style="filter: brightness(0) invert(1);"></div>
                <div class="mc-title">Your Account</div>
            </div>
            <div class="mc-desc">Sign up to track your own spending. Get personalized insights,
            customized budget recommendations, and build better financial habits.</div>
            <ul class="mc-list">
                <li><span class="ck-g">✓</span> Track your actual spending</li>
                <li><span class="ck-g">✓</span> Upload transactions via CSV</li>
                <li><span class="ck-g">✓</span> Personalized recommendations</li>
            </ul>
        </div>
        <div style="height:10px"></div>
        """, unsafe_allow_html=True)
        if st.button("Sign Up / Log In  →", key="login_btn", type="secondary", use_container_width=True):
            st.session_state.mode = 'auth'
            st.rerun()

    st.markdown("""
    <div class="feat-strip">
        <div class="feat-item">
            <div class="feat-icon" style="background:linear-gradient(135deg,#1e3a8a,#3b82f6);"><img src="./app/static/bar-graph.png" width="20" style="filter: brightness(0) invert(1);"></div>
            <div class="feat-name">Smart Analytics</div>
            <div class="feat-desc">AI-powered insights into your spending patterns</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon" style="background:linear-gradient(135deg,#065f46,#10b981);">⚡</div>
            <div class="feat-name">Budget Optimizer</div>
            <div class="feat-desc">Personalized recommendations to reach your savings goals</div>
        </div>
        <div class="feat-item">
            <div class="feat-icon" style="background:linear-gradient(135deg,#9a3412,#ea580c);">🔍</div>
            <div class="feat-name">Anomaly Detection</div>
            <div class="feat-desc">Automatic alerts for unusual spending patterns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DEMO PAGE
# ─────────────────────────────────────────────
def demo_page():

    @st.cache_data
    def load_demo_data():
        try:
            df = pd.read_csv('data/user_monthly_spending.csv', index_col=0)
            for col in CATEGORIES:
                if col not in df.columns:
                    df[col] = 0.0
            return df[CATEGORIES].fillna(0)
        except:
            data = {
                'Sarah Chen':      [320,80,180,220,420,380,280,120,450,90,60,70,85,40],
                'James Mitchell':  [410,50,90,180,280,290,320,200,380,150,40,80,60,220],
                'Emily Rodriguez': [280,120,260,310,190,210,190,95,300,60,90,110,100,30],
            }
            return pd.DataFrame(data, index=CATEGORIES).T

    demo_data = load_demo_data()
    users = demo_data.index.tolist()  # All users, no limit

    if st.session_state.demo_user not in users:
        st.session_state.demo_user = users[0]

    # ── Top bar ──
    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("← Back", key="back_home", type="secondary"):
            st.session_state.mode = None
            st.session_state.demo_user = None
            st.rerun()
    with col_title:
        st.markdown('<div class="page-header">Demo Mode</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(51,65,85,0.5);margin:10px 0 18px;'>", unsafe_allow_html=True)

    # ── User selector (Dropdown) ──
    st.markdown('<div class="section-label">Select a sample user</div>', unsafe_allow_html=True)
    
    selected_user = st.selectbox(
        "Choose a user ID",
        options=users,
        index=users.index(st.session_state.demo_user) if st.session_state.demo_user in users else 0,
        format_func=lambda x: f"User {x}",
        label_visibility="collapsed"
    )
    
    if selected_user != st.session_state.demo_user:
        st.session_state.demo_user = selected_user
        st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    spending = demo_data.loc[selected_user].to_dict()
    total = sum(spending.values())

    # ── Total banner ──
    st.markdown(f"""
    <div class="spend-banner">
        <div class="spend-label">Total Monthly Spending</div>
        <div class="spend-amount">${total:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Chart + Profile ──
    col_chart, col_profile = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="a-card-title"><img src="./app/static/bar-graph.png" width="20" style="vertical-align: middle; margin-right: 6px;"> Spending Breakdown</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        chart_df = pd.DataFrame([
            {'Category': CATEGORY_LABELS[c], 'Amount': spending[c]}
            for c in CATEGORIES if spending[c] > 0
        ]).sort_values('Amount', ascending=True)

        try:
            import plotly.express as px
            fig = px.bar(chart_df, x='Amount', y='Category', orientation='h',
                         color_discrete_sequence=['#3b82f6'],
                         hover_data={'Category': True, 'Amount': ':$,.0f'})
            fig.update_layout(
                paper_bgcolor='rgba(15,23,42,0.9)', plot_bgcolor='rgba(15,23,42,0.9)',
                font=dict(color='#94a3b8', size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    gridcolor='rgba(51,65,85,0.4)',
                    tickfont=dict(color='#64748b'),
                    title=None,
                ),
                yaxis=dict(
                    tickfont=dict(color='#cbd5e1', size=11),
                    title=None,
                ),
                showlegend=False, height=390,
            )
            fig.update_traces(
                marker_line_width=0,
                hovertemplate='<b>%{y}</b><br>$%{x:,.0f}<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except ImportError:
            st.bar_chart(chart_df.set_index('Category')['Amount'])

    with col_profile:
        _, cluster_name = get_user_cluster(spending)
        desc_map = {
            'Average':          "You have typical spending patterns across categories.",
            'High Spender':     "You spend more than average across most categories.",
            'Inconsistent':     "Your spending varies significantly with occasional large purchases.",
            'Budget-Conscious': "You're careful with spending and focus on essentials."
        }
        st.markdown(f"""
        <div class="a-card">
            <div class="a-card-title">👤 Your Profile</div>
            <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">Spending Type</div>
            <div class="cluster-name">{cluster_name}</div>
            <div class="cluster-desc">{desc_map.get(cluster_name,'')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Budget Optimizer ──
    st.markdown("""
    <div class="section-card">
        <div class="a-card-title">⚙️ Budget Optimizer</div>
        <div style="font-size:0.82rem;color:#64748b;margin-bottom:4px;">How much do you want to save?</div>
    </div>
    """, unsafe_allow_html=True)

    max_save = max(200, int(total * 0.6))
    capped_default = min(st.session_state.savings_goal, min(max_save, 3000))
    savings_goal = st.slider(
        "savings", min_value=100, max_value=min(max_save, 3000),
        value=capped_default, step=50,
        label_visibility="collapsed", key="savings_slider"
    )
    st.session_state.savings_goal = savings_goal
    st.markdown(f'<div class="savings-val">${savings_goal:,}</div>', unsafe_allow_html=True)

    recs = get_budget_recommendations(spending, savings_goal)
    if recs:
        st.markdown('<div class="a-card-title" style="margin-bottom:10px;">🎯 Recommended Budget Cuts</div>', unsafe_allow_html=True)
        for r in recs[:6]:
            st.markdown(f"""
            <div class="rec-row">
                <div class="rec-name">{CATEGORY_LABELS[r['category']]}</div>
                <div class="rec-right">
                    <div class="rec-amts">${r['current']:.2f} → ${r['recommended']:.2f}</div>
                    <div class="rec-save">Save ${r['savings']:.2f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="a-card-title" style="margin-bottom:4px;">📋 Priority Order — Easiest to Hardest</div>', unsafe_allow_html=True)
        priorities = get_priority_advice(spending)
        rec_cats = {r['category'] for r in recs}
        filtered_priorities = [(c, d) for c, d in priorities if c in rec_cats]
        for i, (cat, diff) in enumerate(filtered_priorities):
            cls = f"d-{diff.lower()}"
            st.markdown(f"""
            <div class="pri-row">
                <div class="pri-rank">{i+1}</div>
                <div class="pri-cat">{CATEGORY_LABELS[cat]}</div>
                <span class="diff-badge {cls}">{diff}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#64748b;font-size:0.84rem;padding:8px 0;">Savings goal exceeds reducible spending. Try a lower target.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Anomalies ──
    anomalies = detect_anomalies(spending)
    anom_rows_html = ""
    for cat, reason in anomalies:
        anom_rows_html += f"""
        <div class="anom-row">
            <div class="anom-dot"></div>
            <div class="anom-text"><strong>{CATEGORY_LABELS[cat]}</strong>: {reason}</div>
        </div>"""

    if anomalies:
        st.markdown(f"""
        <div class="a-card">
            <div class="anom-header">
                <span style="font-size:16px;">⚠️</span>
                <span class="anom-title">Spending Anomalies</span>
            </div>
            <div class="anom-sub">Found {len(anomalies)} unusual spending pattern{"s" if len(anomalies)>1 else ""}:</div>
            {anom_rows_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="a-card">
            <div class="anom-header">
                <span style="font-size:16px;">⚠️</span>
                <span class="anom-title">Spending Anomalies</span>
            </div>
            <div class="no-anom">✓ No unusual spending patterns detected.</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTH / USER DASHBOARD
# ─────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=130)
        if st.session_state.user:
            st.markdown(f"**{st.session_state.user.email}**")
            if st.button("Logout"):
                logout()
                st.session_state.mode = None
                st.rerun()
        else:
            if st.button("← Back to Home"):
                st.session_state.mode = None
                st.rerun()


def auth_page():
    # Back button
    st.markdown('<div class="back-pill">', unsafe_allow_html=True)
    col_b, _ = st.columns([1, 9])
    with col_b:
        if st.button("← Back", key="auth_back"):
            st.session_state.mode = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login_page(get_supabase())
    with tab2:
        signup_page(get_supabase())


def user_dashboard():
    supabase = get_supabase()
    user_id = st.session_state.user.id
    
    # Top bar with Back and Logout buttons
    col_back, col_title, col_logout = st.columns([1, 7, 1])
    with col_back:
        if st.button("← Back", key="user_back", type="secondary"):
            logout()
            st.session_state.mode = None
            st.rerun()
    with col_title:
        st.markdown(f'<div class="page-header">Welcome, {st.session_state.user.email}</div>', unsafe_allow_html=True)
    with col_logout:
        if st.button("Logout", key="user_logout", type="secondary"):
            logout()
            st.session_state.mode = None
            st.rerun()
    
    st.markdown("<hr style='border:none;border-top:1px solid rgba(51,65,85,0.5);margin:10px 0 18px;'>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Enter Spending", "View Analysis", "History"])
    with tab1:
        enter_spending(supabase, user_id)
    with tab2:
        view_analysis(supabase, user_id)
    with tab3:
        view_history(supabase, user_id)


def enter_spending(supabase, user_id):
    st.markdown("### Enter Your Monthly Spending")
    month = st.text_input("Month (YYYY-MM):", value=datetime.now().strftime("%Y-%m"))
    method = st.radio("Input method:", ["Manual Entry", "Upload Transactions (CSV)"], horizontal=True)

    if method == "Manual Entry":
        spending = {}
        cols = st.columns(2)
        for i, cat in enumerate(CATEGORIES):
            with cols[i % 2]:
                spending[cat] = st.number_input(CATEGORY_LABELS[cat], min_value=0.0, value=0.0, step=10.0, key=f"inp_{cat}")
        if st.button("Save Spending", type="primary"):
            save_spending(supabase, user_id, month, spending)
    else:
        if not RF_LOADED:
            st.warning("Random Forest model not loaded.")
            return
        st.info("**Required columns:** merchant, amt\n\n**Optional columns:** lat, long, city_pop, merch_lat, merch_long, hour, day_of_week, month\n\n*If location columns are missing, defaults will be used.*")
        uploaded = st.file_uploader("Choose CSV", type="csv")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.dataframe(df.head(), use_container_width=True)
                if st.button("Categorize & Save", type="primary"):
                    with st.spinner("Categorizing..."):
                        df_cat = categorize_transactions(df)
                        st.dataframe(df_cat[['merchant','amt','predicted_category']].head(10), use_container_width=True)
                        spending = summarize_by_category(df_cat)
                        save_spending(supabase, user_id, month, spending)
            except Exception as e:
                st.error(f"Error: {e}")


def save_spending(supabase, user_id, month, spending):
    try:
        supabase.table('monthly_spending').upsert(
            {'user_id': user_id, 'month': month, **spending},
            on_conflict='user_id,month'
        ).execute()
        st.success(f"Saved spending for {month}!")
    except Exception as e:
        st.error(f"Error saving: {e}")


def view_analysis(supabase, user_id):
    st.markdown("### Your Analysis")
    
    # Get all months for this user
    resp = supabase.table('monthly_spending').select('*').eq('user_id', user_id)\
        .order('month', desc=True).execute()
    
    if not resp.data:
        st.warning("No spending data found. Enter your spending first.")
        return
    
    # Create month selector
    months = [row['month'] for row in resp.data]
    selected_month = st.selectbox("Select Month", months, index=0)
    
    # Get data for selected month
    month_data = next((row for row in resp.data if row['month'] == selected_month), resp.data[0])
    spending = {cat: float(month_data[cat]) for cat in CATEGORIES}
    total = sum(spending.values())

    st.markdown(f"""
    <div class="spend-banner">
        <div class="spend-label">Total Monthly Spending — {selected_month}</div>
        <div class="spend-amount">${total:,.2f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Chart + Profile ──
    col_chart, col_profile = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown("""
        <div class="a-card">
            <div class="a-card-title"><img src="./app/static/spending-breakdown.png" width="20" style="vertical-align: middle; margin-right: 6px;"> Spending Breakdown</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        chart_df = pd.DataFrame([
            {'Category': CATEGORY_LABELS[c], 'Amount': spending[c]}
            for c in CATEGORIES if spending[c] > 0
        ]).sort_values('Amount', ascending=True)

        try:
            import plotly.express as px
            fig = px.bar(chart_df, x='Amount', y='Category', orientation='h',
                         color_discrete_sequence=['#3b82f6'])
            fig.update_layout(
                paper_bgcolor='rgba(15,23,42,0.9)', plot_bgcolor='rgba(15,23,42,0.9)',
                font=dict(color='#94a3b8', size=11),
                margin=dict(l=10, r=20, t=10, b=10),
                xaxis=dict(gridcolor='rgba(51,65,85,0.4)', tickfont=dict(color='#64748b'), title=None),
                yaxis=dict(tickfont=dict(color='#cbd5e1', size=11), title=None),
                showlegend=False, height=390,
                hoverlabel=dict(bgcolor='rgba(15,23,42,0.95)', bordercolor='#3b82f6', font=dict(color='#f1f5f9', size=13))
            )
            fig.update_traces(marker_line_width=0, hovertemplate='<b>%{y}</b><br>amount : $%{x:,.2f}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except ImportError:
            st.bar_chart(chart_df.set_index('Category')['Amount'])

    with col_profile:
        _, cluster_name = get_user_cluster(spending)
        desc_map = {
            'Average': "You have typical spending patterns across categories.",
            'High Spender': "You spend more than average across most categories.",
            'Inconsistent': "Your spending varies significantly with occasional large purchases.",
            'Budget-Conscious': "You're careful with spending and focus on essentials."
        }
        st.markdown(f"""
        <div class="a-card">
            <div class="a-card-title"><img src="./app/static/login.png" width="20" style="vertical-align: middle; margin-right: 6px;"> Your Profile</div>
            <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">Spending Type</div>
            <div class="cluster-name">{cluster_name}</div>
            <div class="cluster-desc">{desc_map.get(cluster_name,'')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Budget Optimizer with Slider ──
    st.markdown("""
    <div class="section-card">
        <div class="a-card-title"><img src="./app/static/target.png" width="20" style="vertical-align: middle; margin-right: 6px;"> Budget Optimizer</div>
        <div style="font-size:0.82rem;color:#64748b;margin-bottom:4px;">How much do you want to save?</div>
    </div>
    """, unsafe_allow_html=True)

    max_save = max(200, int(total * 0.6))
    capped_default = min(st.session_state.savings_goal, min(max_save, 3000))
    savings_goal = st.slider(
        "savings", min_value=100, max_value=min(max_save, 3000),
        value=capped_default, step=50,
        label_visibility="collapsed", key="user_savings_slider"
    )
    st.session_state.savings_goal = savings_goal
    st.markdown(f'<div class="savings-val">${savings_goal:,}</div>', unsafe_allow_html=True)

    # Get difficulty ratings
    priorities = get_priority_advice(spending)
    difficulty_map = {cat: diff for cat, diff in priorities}

    recs = get_budget_recommendations(spending, savings_goal)
    if recs:
        # Sort by difficulty
        def diff_order(r):
            diff = difficulty_map.get(r['category'], 'Medium')
            return {'Easy': 0, 'Medium': 1, 'Hard': 2}.get(diff, 1)
        sorted_recs = sorted(recs, key=diff_order)

        st.markdown('<div class="a-card-title" style="margin-top:16px;margin-bottom:10px;"><img src="./app/static/target.png" width="20" style="vertical-align: middle; margin-right: 6px;"> Recommended Budget Cuts</div>', unsafe_allow_html=True)
        for r in sorted_recs[:10]:
            diff = difficulty_map.get(r['category'], 'Medium')
            cls = f"d-{diff.lower()}"
            st.markdown(f"""
            <div class="rec-row">
                <div class="rec-left">
                    <div class="rec-name">{CATEGORY_LABELS[r['category']]}</div>
                    <span class="diff-badge {cls}">{diff}</span>
                </div>
                <div class="rec-right">
                    <div class="rec-amts">${r['current']:.2f} → ${r['recommended']:.2f}</div>
                    <div class="rec-save">Save ${r['savings']:.2f}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#64748b;font-size:0.84rem;padding:8px 0;">Savings goal exceeds reducible spending. Try a lower target.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Anomalies ──
    anomalies = detect_anomalies(spending)
    if anomalies:
        anom_rows_html = ""
        for cat, reason in anomalies:
            anom_rows_html += f'<div class="anom-row"><div class="anom-dot"></div><div class="anom-text"><strong>{CATEGORY_LABELS[cat]}</strong>: {reason}</div></div>'
        st.markdown(f"""<div class="a-card">
            <div class="anom-header">
                <span style="font-size:16px;">⚠️</span>
                <span class="anom-title">Spending Anomalies</span>
            </div>
            <div class="anom-sub">Found {len(anomalies)} unusual spending pattern{"s" if len(anomalies)>1 else ""}:</div>
            {anom_rows_html}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="a-card">
            <div class="anom-header">
                <span style="font-size:16px;">⚠️</span>
                <span class="anom-title">Spending Anomalies</span>
            </div>
            <div class="no-anom">✓ No unusual spending patterns detected.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Predict Next Month ──
    if len(resp.data) >= 2:
        predicted = predict_next_month(spending)
        if predicted:
            diff = predicted - total
            diff_pct = (diff / total * 100) if total > 0 else 0
            trend_color = "#ef4444" if diff > 0 else "#10b981"
            trend_icon = '<img src="./app/static/bar-graph.png" width="16" style="vertical-align: middle;">' if diff > 0 else '<img src="./app/static/bar-graph.png" width="16" style="vertical-align: middle; transform: scaleY(-1);">'
            trend_text = f"+${diff:,.2f} ({diff_pct:+.1f}%)" if diff > 0 else f"-${abs(diff):,.2f} ({diff_pct:.1f}%)"
            
            st.markdown(f"""
            <div class="a-card">
                <div class="a-card-title"><img src="./app/static/bar-graph.png" width="20" style="vertical-align: middle; margin-right: 6px;"> Spending Forecast</div>
                <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;">Predicted Next Month</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:1.85rem;font-weight:700;color:#3b82f6;margin-bottom:8px;">${predicted:,.2f}</div>
                <div style="font-size:0.85rem;color:{trend_color};display:flex;align-items:center;gap:6px;">
                    <span>{trend_icon}</span>
                    <span>{trend_text} from current month</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:16px;padding:12px;background:rgba(30,41,59,0.5);border-radius:8px;color:#64748b;font-size:0.85rem;">
            <img src="./app/static/bar-graph.png" width="16" style="vertical-align: middle; margin-right: 4px;"> <strong>Spending Forecast</strong> will be available after 2 months of data.
        </div>
        """, unsafe_allow_html=True)


def view_history(supabase, user_id):
    st.markdown("### Spending History")
    resp = supabase.table('monthly_spending').select('*').eq('user_id', user_id)\
        .order('month', desc=True).execute()
    if not resp.data:
        st.warning("No history found.")
        return
    df = pd.DataFrame(resp.data)
    cols = [c for c in ['month'] + CATEGORIES if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)
    if len(df) > 1:
        df['total'] = df[CATEGORIES].sum(axis=1)
        st.line_chart(df[['month','total']].set_index('month').sort_index())


if __name__ == "__main__":
    main()