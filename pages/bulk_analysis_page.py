"""
Bulk Feedback Analysis Page.
"""

import io
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment_analysis import predict_bulk
from database import insert_bulk_feedback

SAMPLE_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_customer_reviews.csv")


def render_bulk_analysis_page():
    st.markdown("""
        <div class="brand-banner">
            <div class="brand-title-group">
                <h1>Bulk Customer Feedback Analysis</h1>
                <p>Upload CSV datasets of customer feedback to perform high-throughput sentiment classification.</p>
            </div>
            <div class="brand-badge">Batch Processing</div>
        </div>
    """, unsafe_allow_html=True)

    # Download Template Helper
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("<p style='font-size: 0.88rem; color: #64748B;'>Upload any CSV containing a column with customer feedback text. Don't have a dataset ready? Download our ready-to-test 25-review sample CSV:</p>", unsafe_allow_html=True)
    with col_t2:
        if os.path.exists(SAMPLE_CSV_PATH):
            with open(SAMPLE_CSV_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=f,
                    file_name="sample_customer_reviews.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded '{uploaded_file.name}' ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            st.error(f"❌ Failed to parse CSV file: {str(e)}")
            return

        # Column selection
        potential_text_cols = [c for c in df.columns if any(k in c.lower() for k in ["review", "feedback", "text", "comment", "message", "body"])]
        default_index = df.columns.get_loc(potential_text_cols[0]) if potential_text_cols else 0

        col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
        with col_c1:
            text_column = st.selectbox("Select Review Text Column:", df.columns, index=default_index)
        with col_c2:
            id_cols = ["None"] + list(df.columns)
            potential_id_cols = [c for c in df.columns if any(k in c.lower() for k in ["id", "customer", "user"])]
            id_index = id_cols.index(potential_id_cols[0]) if potential_id_cols else 0
            id_column = st.selectbox("Select Customer ID Column (Optional):", id_cols, index=id_index)
        with col_c3:
            cat_cols = ["None"] + list(df.columns)
            potential_cat_cols = [c for c in df.columns if any(k in c.lower() for k in ["category", "dept", "department", "type"])]
            cat_index = cat_cols.index(potential_cat_cols[0]) if potential_cat_cols else 0
            cat_column = st.selectbox("Select Category Column (Optional):", cat_cols, index=cat_index)

        if st.button("⚡ Run Bulk Sentiment Analysis", use_container_width=True):
            with st.spinner(f"Analyzing {len(df)} reviews using TF-IDF + Logistic Regression..."):
                analyzed_df = predict_bulk(
                    df=df,
                    text_column=text_column,
                    id_column=None if id_column == "None" else id_column,
                    category_column=None if cat_column == "None" else cat_column
                )
                st.session_state["analyzed_df"] = analyzed_df
                st.session_state["analyzed_text_col"] = text_column
                st.session_state["analyzed_id_col"] = id_column
                st.session_state["analyzed_cat_col"] = cat_column

    # Render results if available in session state
    if "analyzed_df" in st.session_state:
        res_df = st.session_state["analyzed_df"]
        text_col = st.session_state["analyzed_text_col"]
        id_col = st.session_state["analyzed_id_col"]
        cat_col = st.session_state["analyzed_cat_col"]

        total = len(res_df)
        counts = res_df["Predicted_Sentiment"].value_counts().to_dict()
        pos_count = counts.get("Positive", 0)
        neg_count = counts.get("Negative", 0)
        neu_count = counts.get("Neutral", 0)

        pos_pct = round((pos_count / total) * 100, 1) if total > 0 else 0.0
        neg_pct = round((neg_count / total) * 100, 1) if total > 0 else 0.0
        neu_pct = round((neu_count / total) * 100, 1) if total > 0 else 0.0

        st.markdown("---")
        st.markdown("### 📊 Batch Analysis Summary")

        # Summary KPIs
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-header">
                        <span class="kpi-title">Batch Total</span>
                        <span class="kpi-icon">📦</span>
                    </div>
                    <div class="kpi-value">{total}</div>
                    <div class="kpi-subtext">Processed reviews</div>
                </div>
                <div class="kpi-card positive">
                    <div class="kpi-header">
                        <span class="kpi-title">Positive</span>
                        <span class="kpi-icon">😊</span>
                    </div>
                    <div class="kpi-value" style="color: #059669;">{pos_count}</div>
                    <div class="kpi-subtext">{pos_pct}% of batch</div>
                </div>
                <div class="kpi-card negative">
                    <div class="kpi-header">
                        <span class="kpi-title">Negative</span>
                        <span class="kpi-icon">😡</span>
                    </div>
                    <div class="kpi-value" style="color: #DC2626;">{neg_count}</div>
                    <div class="kpi-subtext">{neg_pct}% of batch</div>
                </div>
                <div class="kpi-card neutral">
                    <div class="kpi-header">
                        <span class="kpi-title">Neutral</span>
                        <span class="kpi-icon">😐</span>
                    </div>
                    <div class="kpi-value" style="color: #4F46E5;">{neu_count}</div>
                    <div class="kpi-subtext">{neu_pct}% of batch</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Batch Visualizations
        col_g1, col_g2 = st.columns([1, 1], gap="medium")
        color_map = {"Positive": "#059669", "Negative": "#DC2626", "Neutral": "#4F46E5"}

        with col_g1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="glass-panel-title"><span>🥧</span> Batch Sentiment Share</div>', unsafe_allow_html=True)
            pie_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Count": [pos_count, neg_count, neu_count]
            })
            fig_p = px.pie(pie_data, values="Count", names="Sentiment", hole=0.5, color="Sentiment", color_discrete_map=color_map)
            fig_p.update_traces(textposition="inside", textinfo="percent+label")
            fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155"), margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_g2:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="glass-panel-title"><span>📊</span> Batch Confidence Distribution</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                res_df,
                x="Confidence_%",
                color="Predicted_Sentiment",
                nbins=15,
                color_discrete_map=color_map,
                labels={"Confidence_%": "Model Confidence (%)"}
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155"), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#E2E8F0"))
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Interactive Table with Filter
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="glass-panel-title"><span>📋</span> Detailed Analyzed Records</div>', unsafe_allow_html=True)

        filter_choice = st.radio("Filter table by sentiment:", ["All", "Positive", "Negative", "Neutral"], horizontal=True)
        display_df = res_df if filter_choice == "All" else res_df[res_df["Predicted_Sentiment"] == filter_choice]

        # Prioritize key presentation columns
        lead_cols = ["Predicted_Sentiment", "Confidence_%", text_col]
        other_cols = [c for c in display_df.columns if c not in lead_cols and c not in ["Cleaned_Text"]]
        st.dataframe(
            display_df[lead_cols + other_cols],
            use_container_width=True,
            height=350
        )

        # Action Buttons: Download CSV & Save to Database
        btn_c1, btn_c2 = st.columns([1, 1])

        with btn_c1:
            csv_buffer = io.StringIO()
            res_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Analyzed CSV Report",
                data=csv_buffer.getvalue(),
                file_name="analyzed_customer_feedback.csv",
                mime="text/csv",
                use_container_width=True
            )

        with btn_c2:
            if st.button("💾 Save Batch Results into Database", use_container_width=True):
                records = []
                for _, row in res_df.iterrows():
                    cid = str(row[id_col]) if id_col and id_col != "None" and id_col in row else None
                    cat = str(row[cat_col]) if cat_col and cat_col != "None" and cat_col in row else "General"
                    records.append({
                        "review_text": str(row[text_col]),
                        "cleaned_text": str(row.get("Cleaned_Text", "")),
                        "sentiment": str(row["Predicted_Sentiment"]),
                        "confidence": float(row["Confidence"]),
                        "prob_positive": float(row.get("Prob_Positive", 0.0)),
                        "prob_negative": float(row.get("Prob_Negative", 0.0)),
                        "prob_neutral": float(row.get("Prob_Neutral", 0.0)),
                        "category": cat,
                        "source": "bulk",
                        "customer_id": cid
                    })
                saved_count = insert_bulk_feedback(records)
                st.success(f"🎉 Successfully persisted {saved_count} batch records into the SQLite database!")

        st.markdown('</div>', unsafe_allow_html=True)

