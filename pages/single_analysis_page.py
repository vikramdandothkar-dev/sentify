"""
Single Feedback Sentiment Analysis Page.
"""

import streamlit as st
from sentiment_analysis import predict_sentiment
from database import insert_feedback


def render_single_analysis_page():
    st.markdown("""
        <div class="brand-banner">
            <div class="brand-title-group">
                <h1>Live Sentiment Analyzer</h1>
                <p>Input customer reviews to classify sentiment, evaluate model confidence, and inspect linguistic features.</p>
            </div>
            <div class="brand-badge">NLP Inference</div>
        </div>
    """, unsafe_allow_html=True)

    # Quick Sample Presets
    st.markdown("<p style='font-size: 0.85rem; color: #64748B; margin-bottom: 0.4rem;'>💡 <b>Quick Test Presets:</b> Click any sample below to load into the text area:</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    preset_pos = "The product quality is absolutely stunning! The shipping was super fast and customer support resolved my setup question in minutes."
    preset_neg = "Terrible experience. The hardware died after just two days and the support team was rude and refused my refund request."
    preset_neu = "The package was delivered on Tuesday afternoon in a standard box. Everything matches the specifications in the manual."
    preset_tricky = "I was worried it might not be worth the price, but it turns out not bad at all and definitely not disappointing."

    if "feedback_input" not in st.session_state:
        st.session_state["feedback_input"] = ""

    with c1:
        if st.button("🌟 Positive Sample", use_container_width=True, help="Load an enthusiastic positive review"):
            st.session_state["feedback_input"] = preset_pos
    with c2:
        if st.button("⚠️ Negative Sample", use_container_width=True, help="Load a critical negative complaint"):
            st.session_state["feedback_input"] = preset_neg
    with c3:
        if st.button("⚖️ Neutral Sample", use_container_width=True, help="Load a factual neutral review"):
            st.session_state["feedback_input"] = preset_neu
    with c4:
        if st.button("🔍 Negation Test", use_container_width=True, help="Test negation handling (e.g. 'not bad', 'not disappointing')"):
            st.session_state["feedback_input"] = preset_tricky

    # Input Form
    with st.form("sentiment_form"):
        user_text = st.text_area(
            "Customer Review / Feedback Text:",
            value=st.session_state.get("feedback_input", ""),
            height=130,
            placeholder="Type or paste customer feedback here... (e.g. 'Loved the seamless experience, will definitely order again!')"
        )

        col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
        with col_opt1:
            category = st.selectbox(
                "Feedback Category:",
                ["General", "Product Quality", "Customer Service", "Delivery & Shipping", "Pricing & Billing", "User Experience"]
            )
        with col_opt2:
            customer_id = st.text_input("Customer ID (Optional):", placeholder="e.g. CUST-8821")
        with col_opt3:
            save_to_db = st.checkbox("Save result to SQLite Database", value=True)

        submit_btn = st.form_submit_button("🚀 Analyze Sentiment", use_container_width=True)

    if submit_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter feedback text before clicking Analyze.")
            return

        with st.spinner("Analyzing text using TF-IDF + Logistic Regression..."):
            result = predict_sentiment(user_text)

        if "error" in result:
            st.error(result["error"])
            return

        sentiment = result["sentiment"]
        conf_pct = result["confidence_percentage"]
        style = result["style"]
        stats = result["text_statistics"]
        tokens = result["token_explanations"]

        # Persist to DB if checked
        if save_to_db:
            inserted_id = insert_feedback(
                review_text=user_text,
                sentiment=sentiment,
                confidence=result["confidence"],
                cleaned_text=result["cleaned_text"],
                prob_pos=result["prob_positive"],
                prob_neg=result["prob_negative"],
                prob_neu=result["prob_neutral"],
                category=category,
                source="single",
                customer_id=customer_id if customer_id.strip() else None
            )
            st.toast(f"✅ Saved to database as record #{inserted_id}", icon="💾")

        # 1. Main Showcase Result Card
        st.markdown(f"""
            <div class="sentiment-result-card" style="border-left: 5px solid {style['color']};">
                <div class="sentiment-result-header">
                    <div class="sentiment-label-group">
                        <div class="sentiment-emoji-large">{style['icon']}</div>
                        <div>
                            <div class="sentiment-name" style="color: {style['color']};">{sentiment} Sentiment</div>
                            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">{style['label_description']}</div>
                        </div>
                    </div>
                    <div class="sentiment-confidence-badge" style="background: {style['bg_color']}; color: {style['color']}; border: 1px solid {style['border_color']};">
                        {conf_pct}% Confidence
                    </div>
                </div>
                <div style="font-size: 0.95rem; color: #1E293B; font-style: italic; background: #F8FAFC; padding: 0.9rem; border-radius: 8px; border: 1px solid #E2E8F0;">
                    "{user_text}"
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 2. Detailed Probability Breakdown and Linguistics
        col_res1, col_res2 = st.columns([1, 1], gap="medium")

        with col_res1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="glass-panel-title"><span>📊</span> Class Probability Distribution</div>', unsafe_allow_html=True)

            probs = result["probabilities"]
            p_pos = probs.get("Positive", 0.0) * 100
            p_neg = probs.get("Negative", 0.0) * 100
            p_neu = probs.get("Neutral", 0.0) * 100

            st.write(f"**Positive:** {p_pos:.1f}%")
            st.progress(min(max(p_pos / 100.0, 0.0), 1.0))

            st.write(f"**Negative:** {p_neg:.1f}%")
            st.progress(min(max(p_neg / 100.0, 0.0), 1.0))

            st.write(f"**Neutral:** {p_neu:.1f}%")
            st.progress(min(max(p_neu / 100.0, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res2:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="glass-panel-title"><span>🧠</span> Explainable AI (XAI) Feature Importance</div>', unsafe_allow_html=True)

            if tokens:
                st.markdown("<p style='font-size: 0.82rem; color: #64748B;'>Words with the strongest model coefficient impact:</p>", unsafe_allow_html=True)
                token_html = ""
                for tok in tokens:
                    tok_cls = "token-positive" if tok["importance"] > 0 else "token-negative"
                    impact_symbol = "+" if tok["importance"] > 0 else ""
                    token_html += f'<span class="token-pill {tok_cls}">{tok["word"]} ({impact_symbol}{tok["importance"]})</span> '
                st.markdown(token_html, unsafe_allow_html=True)
            else:
                st.write("Neutral / baseline vocabulary detected.")

            st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 1rem 0;'>", unsafe_allow_html=True)
            st.markdown('<div class="glass-panel-title" style="font-size: 0.95rem;"><span>📏</span> Text Statistics</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style="font-size: 0.85rem; color: #334155; display: flex; gap: 1.5rem;">
                    <div><b>Words:</b> {stats['word_count']}</div>
                    <div><b>Characters:</b> {stats['char_count']}</div>
                    <div><b>Avg Word Length:</b> {stats['avg_word_length']}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

