"""
Dashboard Page: High-Level Analytics & KPIs.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from database import get_feedback_statistics, get_all_feedback


def render_dashboard_page():
    st.markdown("""
        <div class="brand-banner">
            <div class="brand-title-group">
                <h1>Executive Sentiment Dashboard</h1>
                <p>Real-time customer satisfaction metrics, sentiment distribution, and operational KPIs.</p>
            </div>
            <div class="brand-badge">Live Analytics</div>
        </div>
    """, unsafe_allow_html=True)

    stats = get_feedback_statistics()

    # If no data exists, prompt user
    if stats["total"] == 0:
        st.info("No customer reviews recorded yet. Analyze some feedback or seed sample data in the History tab.")
        return

    # 1. KPI Cards Row
    total = stats["total"]
    pos = stats["positive"]
    neg = stats["negative"]
    neu = stats["neutral"]
    csat = stats["csat_score"]
    avg_conf = stats["avg_confidence"]

    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Total Reviews</span>
                    <span class="kpi-icon">📝</span>
                </div>
                <div class="kpi-value">{total}</div>
                <div class="kpi-subtext">Aggregated customer feedback</div>
            </div>
            <div class="kpi-card positive">
                <div class="kpi-header">
                    <span class="kpi-title">Positive</span>
                    <span class="kpi-icon">😊</span>
                </div>
                <div class="kpi-value" style="color: #059669;">{pos}</div>
                <div class="kpi-subtext">{stats['positive_pct']}% of total volume</div>
            </div>
            <div class="kpi-card negative">
                <div class="kpi-header">
                    <span class="kpi-title">Negative</span>
                    <span class="kpi-icon">😡</span>
                </div>
                <div class="kpi-value" style="color: #DC2626;">{neg}</div>
                <div class="kpi-subtext">{stats['negative_pct']}% critical attention</div>
            </div>
            <div class="kpi-card neutral">
                <div class="kpi-header">
                    <span class="kpi-title">Neutral</span>
                    <span class="kpi-icon">😐</span>
                </div>
                <div class="kpi-value" style="color: #4F46E5;">{neu}</div>
                <div class="kpi-subtext">{stats['neutral_pct']}% operational/queries</div>
            </div>
            <div class="kpi-card satisfaction">
                <div class="kpi-header">
                    <span class="kpi-title">Satisfaction (CSAT)</span>
                    <span class="kpi-icon">⭐</span>
                </div>
                <div class="kpi-value" style="color: #D97706;">{csat}%</div>
                <div class="kpi-subtext">Avg Confidence: {avg_conf}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Charts Row
    df = get_all_feedback()
    col1, col2 = st.columns([1, 1], gap="medium")

    # Color mapping for sentiment
    color_map = {
        "Positive": "#059669",
        "Negative": "#DC2626",
        "Neutral": "#4F46E5"
    }

    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="glass-panel-title"><span>🥧</span> Sentiment Distribution</div>', unsafe_allow_html=True)
        
        donut_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [pos, neg, neu]
        })
        
        fig_donut = px.pie(
            donut_data,
            values="Count",
            names="Sentiment",
            hole=0.55,
            color="Sentiment",
            color_discrete_map=color_map
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
            hoverinfo="label+value+percent"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Plus Jakarta Sans"),
            margin=dict(t=20, b=20, l=10, r=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="glass-panel-title"><span>📊</span> Category vs. Sentiment Breakdown</div>', unsafe_allow_html=True)

        if not df.empty and "category" in df.columns:
            cat_summary = df.groupby(["category", "sentiment"]).size().reset_index(name="Count")
            fig_bar = px.bar(
                cat_summary,
                x="category",
                y="Count",
                color="sentiment",
                barmode="group",
                color_discrete_map=color_map,
                labels={"category": "Review Category", "Count": "Number of Reviews"}
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", family="Plus Jakarta Sans"),
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis=dict(showgrid=False, title=None),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Count"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("No category data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Timeline / Volume Trends
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>📈</span> Sentiment Volume Over Time</div>', unsafe_allow_html=True)

    if not df.empty and "created_at" in df.columns:
        df_trend = df.copy()
        df_trend["date"] = pd.to_datetime(df_trend["created_at"]).dt.date
        trend_summary = df_trend.groupby(["date", "sentiment"]).size().reset_index(name="Count")

        fig_trend = px.line(
            trend_summary,
            x="date",
            y="Count",
            color="sentiment",
            markers=True,
            color_discrete_map=color_map,
            labels={"date": "Analysis Date", "Count": "Review Count"}
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Plus Jakarta Sans"),
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Volume"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Recent Reviews Stream
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>🕒</span> Recent Feedback Activity</div>', unsafe_allow_html=True)

    recent_df = df.head(5)
    for _, row in recent_df.iterrows():
        s = row["sentiment"]
        badge_cls = "badge-positive" if s == "Positive" else ("badge-negative" if s == "Negative" else "badge-neutral")
        icon = "😊" if s == "Positive" else ("😡" if s == "Negative" else "😐")
        conf_pct = round(row["confidence"] * 100, 1)

        st.markdown(f"""
            <div style="padding: 0.85rem 1rem; border-bottom: 1px solid #E2E8F0; display: flex; align-items: center; justify-content: space-between; gap: 1rem;">
                <div style="flex: 1;">
                    <div style="font-size: 0.92rem; color: #0F172A; font-weight: 500;">"{row['review_text']}"</div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.25rem;">
                        <span>👤 {row['customer_id']}</span> &bull; 
                        <span>🏷️ {row['category']}</span> &bull; 
                        <span>📅 {row['created_at']}</span>
                    </div>
                </div>
                <div style="text-align: right; min-width: 140px;">
                    <span class="badge-pill {badge_cls}">{icon} {s}</span>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.25rem; font-family: monospace;">{conf_pct}% Conf</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
