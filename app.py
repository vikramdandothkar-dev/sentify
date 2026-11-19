"""
AI-Based Customer Feedback Sentiment Analysis System
Main Application Entry Point.

College Internship Capstone Project
Tech Stack: Python, Scikit-learn, NLTK, Streamlit, Plotly, SQLite, Pandas
"""

import os
import sys
import warnings

# Suppress benign warnings
warnings.filterwarnings("ignore")

import streamlit as st

# Configure wide responsive layout
st.set_page_config(
    page_title="Sentify AI | Customer Feedback Sentiment System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current workspace to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from database import init_db, seed_sample_data, get_feedback_statistics
from pages.dashboard_page import render_dashboard_page
from pages.single_analysis_page import render_single_analysis_page
from pages.bulk_analysis_page import render_bulk_analysis_page
from pages.history_page import render_history_page
from pages.about_page import render_about_page


def load_custom_css():
    """Injects custom SaaS dark navy and purple CSS stylesheet."""
    css_path = os.path.join(current_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # 1. Initialize Database & Seed initial demo records
    init_db()
    seed_sample_data(force=False)

    # 2. Inject CSS
    load_custom_css()

    # Hide Streamlit's auto-generated multipage navigation buttons
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] { display: none !important; }
            [data-testid="stPageLink"] { display: none !important; }
            section[data-testid="stSidebar"] nav { display: none !important; }
            .st-emotion-cache-1cypcdb { display: none !important; }
            [data-testid="stAppViewBlockContainer"] [data-testid="stPageLink"] { display: none !important; }
            div[data-testid="stBottomBlockContainer"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

    # 3. Sidebar Navigation & Brand
    with st.sidebar:
        st.markdown("""
            <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid rgba(0,0,0,0.08); margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="font-size: 2rem;">🔮</div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #1E293B; letter-spacing: -0.02em;">Sentify AI</div>
                        <div style="font-size: 0.72rem; color: #6366F1; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Sentiment Intelligence</div>
                    </div>
                </div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.6rem; line-height: 1.4;">
                    AI-Based Customer Feedback Sentiment Analysis System
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Navigation</p>", unsafe_allow_html=True)
        
        pages_dict = {
            "📊 Dashboard": render_dashboard_page,
            "💬 Sentiment Analysis": render_single_analysis_page,
            "📁 Bulk Analysis": render_bulk_analysis_page,
            "📜 Feedback History": render_history_page,
            "ℹ️ About Project": render_about_page
        }

        selection = st.radio(
            "Select Page:",
            list(pages_dict.keys()),
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(0,0,0,0.08); margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
        
        # Sidebar Status Pill & Quick KPIs
        stats = get_feedback_statistics()
        st.markdown(f"""
            <div style="background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem; margin-bottom: 1rem;">
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">System Status</div>
                <div style="display: flex; align-items: center; gap: 0.4rem; margin-top: 0.3rem;">
                    <span style="height: 8px; width: 8px; background: #10B981; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.82rem; color: #1E293B; font-weight: 600;">Model Online (TF-IDF + LR)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.4rem; margin-top: 0.25rem;">
                    <span style="height: 8px; width: 8px; background: #10B981; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.82rem; color: #1E293B; font-weight: 600;">Database: SQLite Active</span>
                </div>
                <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 0.6rem 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #475569;">
                    <span>Total Reviews:</span>
                    <b style="color: #1E293B;">{stats['total']}</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #475569; margin-top: 0.2rem;">
                    <span>CSAT Rate:</span>
                    <b style="color: #059669;">{stats['csat_score']}%</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align: center; font-size: 0.72rem; color: #94A3B8; padding-top: 0.5rem;">
                College Internship Project &bull; 2026<br>
                Built with Python, Scikit-learn & Streamlit
            </div>
        """, unsafe_allow_html=True)

    # 4. Render the chosen page
    page_fn = pages_dict.get(selection, render_dashboard_page)
    page_fn()

    # 5. Global Footer
    st.markdown("""
        <div class="app-footer">
            <b>AI-Based Customer Feedback Sentiment Analysis System</b> &bull;
            Powered by TF-IDF & Logistic Regression &bull; Designed with Dark Navy & Purple Theme
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

