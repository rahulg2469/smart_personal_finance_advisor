import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
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

# Supabase setup
SUPABASE_URL = "https://lkcbsdngpbpffnxtnlys.supabase.co"
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")  # Add your anon key to .streamlit/secrets.toml

# Initialize Supabase client
@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Categories
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

# Page config
st.set_page_config(
    page_title="Smart Finance Advisor",
    layout="wide"
)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'mode' not in st.session_state:
    st.session_state.mode = None

def main():
    # Add custom CSS for dark mode gradient, blobs, and cards
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e293b 100%);
        min-height: 100vh;
    }
    
    /* Decorative blobs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -100px;
        right: -100px;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(30, 58, 138, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -100px;
        left: -100px;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(6, 78, 59, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    
    .card {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 1;
    }
    
    .card-demo {
        border-color: #3b82f6;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }
    
    .card-demo:hover {
        border-color: #60a5fa;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.35);
        transform: translateY(-4px);
    }
    
    .card-account {
        border-color: #10b981;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15);
    }
    
    .card-account:hover {
        border-color: #34d399;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.35);
        transform: translateY(-4px);
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 12px;
        color: #ffffff;
    }
    
    .card-desc {
        color: #e2e8f0;
        margin-bottom: 16px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .card-list {
        color: #f1f5f9;
        margin-bottom: 20px;
        padding-left: 0;
    }
    
    .card-list li {
        margin-bottom: 10px;
        list-style: none;
        font-size: 0.95rem;
    }
    
    .center-title {
        text-align: center;
        margin-bottom: 8px;
        color: #f8fafc;
        position: relative;
        z-index: 1;
    }
    
    .center-subtitle {
        text-align: center;
        margin-bottom: 40px;
        color: #ffffff;
        position: relative;
        z-index: 1;
    }
    
    /* Features section */
    .features-section {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 24px 30px;
        margin-top: 30px;
        border: 1px solid #334155;
        position: relative;
        z-index: 1;
    }
    
    .features-title {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 24px;
    }
    
    .feature-card {
        text-align: center;
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px;
        font-size: 22px;
    }
    
    .feature-icon-blue {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    }
    
    .feature-icon-green {
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
    }
    
    .feature-icon-orange {
        background: linear-gradient(135deg, #9a3412 0%, #ea580c 100%);
    }
    
    .feature-name {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 6px;
        font-size: 0.95rem;
    }
    
    .feature-desc {
        color: #94a3b8;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    /* Primary button styling (blue for demo) */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        height: 48px;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* Secondary button styling (green for login) */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        height: 48px;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.4);
        transform: translateY(-2px);
    }
    
    /* Match card heights */
    .card {
        min-height: 280px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Mode selection if not chosen
    if st.session_state.mode is None:
        st.markdown("<h1 class='center-title'><img src='./app/static/logo.jpeg' width='60' style='vertical-align: middle; margin-right: 12px; border-radius: 8px;'> Smart Personal Finance Advisor</h1>", unsafe_allow_html=True)
        st.markdown("<p class='center-subtitle'>Take control of your spending with AI-powered insights. Track your budget, get personalized recommendations, and build better financial habits.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class="card card-demo">
                <div class="card-title"><img src="./app/static/bar-graph.png" width="24" style="vertical-align: middle; margin-right: 8px;"> Demo Mode</div>
                <div class="card-desc">Explore with sample users from our dataset. No signup needed. See real spending patterns and get recommendations instantly.</div>
                <ul class="card-list">
                    <li><img src="./app/static/check.png" width="18" style="vertical-align: middle; margin-right: 8px;"> Browse sample spending data</li>
                    <li><img src="./app/static/check.png" width="18" style="vertical-align: middle; margin-right: 8px;"> View AI-powered analysis</li>
                    <li><img src="./app/static/check.png" width="18" style="vertical-align: middle; margin-right: 8px;"> Get budget recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Demo  ➔", use_container_width=True, key="demo_btn", type="primary"):
                st.session_state.mode = 'demo'
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="card card-account">
                <div class="card-title"><img src="./app/static/login.png" width="24" style="vertical-align: middle; margin-right: 8px;"> Your Account</div>
                <div class="card-desc">Sign up to track your own spending. Get personalized insights, customized budget recommendations, and build better financial habits.</div>
                <ul class="card-list">
                    <li><img src="./app/static/check-green.png" width="18" style="vertical-align: middle; margin-right: 8px;"> Track your actual spending</li>
                    <li><img src="./app/static/check-green.png" width="18" style="vertical-align: middle; margin-right: 8px;"> Upload transactions via CSV</li>
                    <li><img src="./app/static/check-green.png" width="18" style="vertical-align: middle; margin-right: 8px;"> Personalized recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Login / Sign Up  ➔", use_container_width=True, key="login_btn", type="secondary"):
                st.session_state.mode = 'auth'
                st.rerun()
        
        # Features section
        st.markdown("""
        <div class="features-section">
            <div class="features-title">Powerful Features</div>
            <div style="display: flex; justify-content: space-around; align-items: flex-start;">
                <div class="feature-card">
                    <div class="feature-icon feature-icon-blue">📈</div>
                    <div class="feature-name">Smart Analytics</div>
                    <div class="feature-desc">Get insights into your spending patterns with AI-powered analysis</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon feature-icon-green">⚡</div>
                    <div class="feature-name">Budget Optimizer</div>
                    <div class="feature-desc">Get personalized recommendations to reach your savings goals</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon feature-icon-orange">📊</div>
                    <div class="feature-name">Anomaly Detection</div>
                    <div class="feature-desc">Stay alerted to unusual spending patterns automatically</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # Sidebar for navigation
    with st.sidebar:
        if st.session_state.mode == 'demo':
            st.markdown("### Demo Mode")
            if st.button("Back to Home"):
                st.session_state.mode = None
                st.rerun()
        elif st.session_state.user:
            st.markdown(f"### Welcome!")
            st.write(f"Logged in as: {st.session_state.user.email}")
            if st.button("Logout"):
                logout()
                st.session_state.mode = None
                st.rerun()
        else:
            if st.button("Back to Home"):
                st.session_state.mode = None
                st.rerun()
    
    # Route to appropriate page
    if st.session_state.mode == 'demo':
        demo_mode()
    elif st.session_state.mode == 'auth':
        if st.session_state.user:
            user_dashboard()
        else:
            auth_page()

def auth_page():
    """Handle login/signup"""
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        login_page(get_supabase())
    
    with tab2:
        signup_page(get_supabase())

def demo_mode():
    """Demo mode with sample users"""
    st.markdown("### Demo Mode")
    st.info("Select a sample user to see the full analysis")
    
    # Load demo data
    @st.cache_data
    def load_demo_data():
        try:
            df = pd.read_csv('data/user_monthly_spending.csv', index_col=0)
            return df
        except:
            st.error("Demo data not found. Please add user_monthly_spending.csv to the data/ folder.")
            return None
    
    demo_data = load_demo_data()
    
    if demo_data is not None:
        # User selection
        users = demo_data.index.tolist()[:50]  # First 50 users
        selected_user = st.selectbox("Select a user:", users)
        
        if selected_user:
            user_spending = demo_data.loc[selected_user].to_dict()
            show_analysis(user_spending, is_demo=True)

def user_dashboard():
    """Dashboard for logged-in users"""
    supabase = get_supabase()
    user_id = st.session_state.user.id
    
    # Tabs for different actions
    tab1, tab2, tab3 = st.tabs(["Enter Spending", "View Analysis", "History"])
    
    with tab1:
        enter_spending(supabase, user_id)
    
    with tab2:
        view_analysis(supabase, user_id)
    
    with tab3:
        view_history(supabase, user_id)

def enter_spending(supabase, user_id):
    """Form to enter monthly spending - manual or CSV upload"""
    st.markdown("### Enter Your Monthly Spending")
    
    # Month selector
    current_month = datetime.now().strftime("%Y-%m")
    month = st.text_input("Month (YYYY-MM):", value=current_month)
    
    # Toggle between manual and upload
    input_method = st.radio("How would you like to enter your spending?", 
                            ["Manual Entry", "Upload Transactions (CSV)"],
                            horizontal=True)
    
    if input_method == "Manual Entry":
        st.markdown("#### Enter amounts for each category:")
        
        # Create input fields for each category
        spending = {}
        cols = st.columns(2)
        
        for i, cat in enumerate(CATEGORIES):
            with cols[i % 2]:
                spending[cat] = st.number_input(
                    CATEGORY_LABELS[cat],
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                    key=f"input_{cat}"
                )
        
        if st.button("Save Spending", type="primary"):
            save_spending(supabase, user_id, month, spending)
    
    else:  # CSV Upload
        st.markdown("#### Upload your transactions")
        
        if not RF_LOADED:
            st.warning("Random Forest model not loaded. Please add random_forest_model.pkl to models/ folder.")
            return
        
        st.info("CSV should have columns: merchant, amt (amount). Optional: hour, day_of_week")
        
        # Sample CSV format
        with st.expander("See example CSV format"):
            st.code("""merchant,amt,hour,day_of_week
Walmart,45.00,14,2
Netflix,15.99,20,5
Shell Gas Station,40.00,8,1
Whole Foods,120.50,11,6
Amazon,89.99,22,3""")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(df)} transactions")
                
                # Show preview
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Categorize & Save", type="primary"):
                    with st.spinner("Categorizing transactions..."):
                        # Categorize using Random Forest
                        df_categorized = categorize_transactions(df)
                        
                        # Show categorized results
                        st.markdown("#### Categorized Transactions:")
                        st.dataframe(
                            df_categorized[['merchant', 'amt', 'predicted_category']].head(10),
                            use_container_width=True
                        )
                        
                        # Summarize by category
                        spending = summarize_by_category(df_categorized)
                        
                        # Show summary
                        st.markdown("#### Spending Summary:")
                        summary_df = pd.DataFrame([
                            {'Category': CATEGORY_LABELS[cat], 'Amount': f"${amt:,.2f}"}
                            for cat, amt in spending.items() if amt > 0
                        ])
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Save to database
                        save_spending(supabase, user_id, month, spending)
                        
            except Exception as e:
                st.error(f"Error processing CSV: {e}")


def save_spending(supabase, user_id, month, spending):
    """Save spending data to Supabase"""
    data = {
        'user_id': user_id,
        'month': month,
        **spending
    }
    
    try:
        supabase.table('monthly_spending').upsert(data, on_conflict='user_id,month').execute()
        st.success(f"Spending for {month} saved!")
    except Exception as e:
        st.error(f"Error saving: {e}")

def view_analysis(supabase, user_id):
    """View analysis for logged-in user"""
    st.markdown("### Your Analysis")
    
    # Fetch user's spending data
    response = supabase.table('monthly_spending')\
        .select('*')\
        .eq('user_id', user_id)\
        .order('month', desc=True)\
        .limit(1)\
        .execute()
    
    if not response.data:
        st.warning("No spending data found. Please enter your spending first.")
        return
    
    latest = response.data[0]
    user_spending = {cat: float(latest[cat]) for cat in CATEGORIES}
    
    show_analysis(user_spending, is_demo=False, supabase=supabase, user_id=user_id)

def view_history(supabase, user_id):
    """View spending history"""
    st.markdown("### Your Spending History")
    
    response = supabase.table('monthly_spending')\
        .select('*')\
        .eq('user_id', user_id)\
        .order('month', desc=True)\
        .execute()
    
    if not response.data:
        st.warning("No spending history found.")
        return
    
    df = pd.DataFrame(response.data)
    df = df[['month', 'total_spent'] + CATEGORIES]
    
    st.dataframe(df, use_container_width=True)
    
    # Simple chart
    if len(df) > 1:
        chart_data = df[['month', 'total_spent']].set_index('month').sort_index()
        st.line_chart(chart_data)

def show_analysis(spending: dict, is_demo: bool = False, supabase=None, user_id=None):
    """Show full analysis for given spending data"""
    
    total = sum(spending.values())
    st.markdown(f"**Total Monthly Spending: ${total:,.2f}**")
    
    # Spending breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Spending Breakdown")
        breakdown_df = pd.DataFrame([
            {'Category': CATEGORY_LABELS[cat], 'Amount': f"${spending[cat]:,.2f}", 'Percent': f"{spending[cat]/total*100:.1f}%"}
            for cat in CATEGORIES if spending[cat] > 0
        ])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Your Profile")
        cluster, cluster_name = get_user_cluster(spending)
        st.metric("Spending Type", cluster_name)
        
        cluster_descriptions = {
            'Average': "You have typical spending patterns across categories.",
            'High Spender': "You spend more than average across most categories.",
            'Inconsistent': "Your spending varies significantly, with occasional large purchases.",
            'Budget-Conscious': "You're careful with spending and focus on essentials."
        }
        st.info(cluster_descriptions.get(cluster_name, ""))
    
    st.divider()
    
    # Budget Optimizer
    st.markdown("#### Budget Optimizer")
    savings_goal = st.slider("How much do you want to save?", 100, 2000, 500, step=100)
    
    if st.button("Get Recommendations"):
        recommendations = get_budget_recommendations(spending, savings_goal)
        
        if recommendations:
            st.markdown("##### Recommended Budget Cuts:")
            for rec in recommendations[:5]:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{CATEGORY_LABELS[rec['category']]}**")
                with col2:
                    st.write(f"${rec['current']:.2f} → ${rec['recommended']:.2f}")
                with col3:
                    st.write(f"Save ${rec['savings']:.2f}")
            
            # Priority advice
            st.markdown("##### Priority Order (Easiest to Hardest):")
            priorities = get_priority_advice(spending)
            
            priority_df = pd.DataFrame([
                {'Rank': i+1, 'Category': CATEGORY_LABELS[cat], 'Difficulty': diff}
                for i, (cat, diff) in enumerate(priorities)
            ])
            st.dataframe(priority_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Anomaly Detection
    st.markdown("#### Spending Anomalies")
    anomalies = detect_anomalies(spending)
    
    if anomalies:
        st.warning(f"Found {len(anomalies)} unusual spending patterns:")
        for cat, reason in anomalies:
            st.write(f"• **{CATEGORY_LABELS[cat]}**: {reason}")
    else:
        st.success("No unusual spending patterns detected.")

if __name__ == "__main__":
    main()