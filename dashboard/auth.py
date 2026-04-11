import streamlit as st
from supabase import Client

def login_page(supabase: Client):
    """Login form"""
    st.markdown("### Login")
    
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_btn"):
        if not email or not password:
            st.error("Please enter email and password")
            return
        
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            st.session_state.user = response.user
            st.success("Login successful!")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {str(e)}")

def signup_page(supabase: Client):
    """Signup form"""
    st.markdown("### Sign Up")
    
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
    
    if st.button("Sign Up", key="signup_btn"):
        if not email or not password:
            st.error("Please enter email and password")
            return
        
        if password != confirm_password:
            st.error("Passwords don't match")
            return
        
        if len(password) < 6:
            st.error("Password must be at least 6 characters")
            return
        
        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if response.user:
                st.success("Account created! Please check your email to verify, then login.")
            else:
                st.error("Signup failed. Please try again.")
        except Exception as e:
            st.error(f"Signup failed: {str(e)}")

def logout():
    """Clear session"""
    st.session_state.user = None
    st.session_state.mode = None
