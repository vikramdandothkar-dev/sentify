"""
Feedback History & Database Management Page.
"""

import io
from collections import Counter
import streamlit as st
import pandas as pd
import plotly.express as px
from database import (
    get_all_feedback,
    get_feedback_statistics,
    delete_feedback,
    clear_all_feedback,
    seed_sample_data
)
from preprocessing import clean_text


def render_history_page():
    st.markdown("""
        <div class="brand-banner">
            <div class="brand-title-group">
                <h1>Feedback Archive & Historic Analytics</h1>
                <p>Search, filter, inspect, and export all customer reviews persisted in the SQLite database.</p>
            </div>
            <div class="brand-badge">SQLite Storage</div>
        </div>
    """, unsafe_allow_html=True)

    # Filter Controls
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>🔍</span> Filter & Search Records</div>', unsafe_allow_html=True)

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1])

    with f_col1:
        search_kw = st.text_input("Search Text or Customer ID:", placeholder="e.g. support, battery, CUST-1042")
    with f_col2:
        sent_filter = st.selectbox("Sentiment:", ["All", "Positive", "Negative", "Neutral"])
    with f_col3:
        cat_filter = st.selectbox("Category:", ["All", "General", "Product Quality", "Customer Service", "Delivery & Shipping", "Pricing & Billing", "User Experience"])
    with f_col4:
        src_filter = st.selectbox("Source:", ["All", "Single", "Bulk", "Sample"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Fetch records matching criteria
    df = get_all_feedback(
        sentiment_filter=sent_filter,
        category_filter=cat_filter,
        source_filter=src_filter,
        search_keyword=search_kw
    )

    # Action Toolbar
    t_col1, t_col2, t_col3 = st.columns([2, 1, 1])
    with t_col1:
        st.markdown(f"<p style='color: #64748B; font-size: 0.9rem; margin-top: 0.5rem;'>Displaying <b>{len(df)}</b> matching customer review records.</p>", unsafe_allow_html=True)

    with t_col2:
        if not df.empty:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv_buf.getvalue(),
                file_name="sentiment_history_export.csv",
                mime="text/csv",
                use_container_width=True
            )

    with t_col3:
        # Seed or Reset modal/expander
        with st.popover("⚙️ Database Controls"):
            st.markdown("<b>Database Administration</b>", unsafe_allow_html=True)
            if st.button("🌱 Re-Seed Sample Records", use_container_width=True):
                seed_sample_data(force=True)
                st.success("Sample reviews seeded!")
                st.rerun()

            if st.button("🗑️ Clear All Records", use_container_width=True):
                clear_all_feedback()
                st.warning("All records deleted from SQLite.")
                st.rerun()

    if df.empty:
        st.info("No records matched your search query. Try broadening your filter criteria.")
        return

    # Data Table View
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>📜</span> Stored Customer Reviews</div>', unsafe_allow_html=True)

    # Format table for display
    display_df = df.copy()
    display_df["Confidence_%"] = (display_df["confidence"] * 100).round(1)
    
    view_cols = ["id", "sentiment", "Confidence_%", "review_text", "category", "source", "customer_id", "created_at"]
    st.dataframe(
        display_df[view_cols],
        use_container_width=True,
        height=380,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Confidence_%": st.column_config.NumberColumn("Conf (%)", format="%.1f%%", width="small"),
            "review_text": st.column_config.TextColumn("Customer Review Text", width="large"),
            "category": st.column_config.TextColumn("Category", width="medium"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "customer_id": st.column_config.TextColumn("Customer", width="small"),
            "created_at": st.column_config.DatetimeColumn("Timestamp", width="medium")
        }
    )

    # Single Record Deletion Tool
    del_col1, del_col2 = st.columns([3, 1])
    with del_col1:
        del_id = st.number_input("Enter Record ID to Delete:", min_value=1, step=1)
    with del_col2:
        st.write("") # spacing
        st.write("")
        if st.button("❌ Delete Record", use_container_width=True):
            if delete_feedback(int(del_id)):
                st.success(f"Record #{del_id} removed from database.")
                st.rerun()
            else:
                st.error(f"Record #{del_id} not found.")

    st.markdown('</div>', unsafe_allow_html=True)

    # NLP Frequent Keywords / Word Cloud Analytics
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>🔤</span> Top Frequent Customer Vocabulary by Sentiment</div>', unsafe_allow_html=True)

    def extract_top_words(subset_df, top_n=8):
        all_tokens = []
        for text in subset_df["review_text"].dropna():
            cleaned = clean_text(str(text), remove_stopwords=True)
            all_tokens.extend([w for w in cleaned.split() if len(w) > 3])
        return Counter(all_tokens).most_common(top_n)

    kw_col1, kw_col2, kw_col3 = st.columns(3)

    pos_df = df[df["sentiment"] == "Positive"]
    neg_df = df[df["sentiment"] == "Negative"]
    neu_df = df[df["sentiment"] == "Neutral"]

    with kw_col1:
        st.markdown("<b style='color: #34D399;'>😊 Positive Top Words</b>", unsafe_allow_html=True)
        top_pos = extract_top_words(pos_df)
        if top_pos:
            for w, cnt in top_pos:
                st.markdown(f"- `<span style='color: #34D399; font-weight:600;'>{w}</span>` ({cnt} mentions)", unsafe_allow_html=True)
        else:
            st.write("No positive words.")

    with kw_col2:
        st.markdown("<b style='color: #F87171;'>😡 Negative Top Words</b>", unsafe_allow_html=True)
        top_neg = extract_top_words(neg_df)
        if top_neg:
            for w, cnt in top_neg:
                st.markdown(f"- `<span style='color: #F87171; font-weight:600;'>{w}</span>` ({cnt} mentions)", unsafe_allow_html=True)
        else:
            st.write("No negative words.")

    with kw_col3:
        st.markdown("<b style='color: #C084FC;'>😐 Neutral Top Words</b>", unsafe_allow_html=True)
        top_neu = extract_top_words(neu_df)
        if top_neu:
            for w, cnt in top_neu:
                st.markdown(f"- `<span style='color: #C084FC; font-weight:600;'>{w}</span>` ({cnt} mentions)", unsafe_allow_html=True)
        else:
            st.write("No neutral words.")

    st.markdown('</div>', unsafe_allow_html=True)

