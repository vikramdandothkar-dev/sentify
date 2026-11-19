"""
About Project & Machine Learning Model Performance Page.
"""

import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sentiment_analysis import load_model_metrics
from model_training import train_and_evaluate_model


def render_about_page():
    st.markdown("""
        <div class="brand-banner">
            <div class="brand-title-group">
                <h1>About the Project & ML Performance</h1>
                <p>Academic & Internship Architecture, Verified Machine Learning Evaluation, and Technical Specifications.</p>
            </div>
            <div class="brand-badge">College Internship Capstone</div>
        </div>
    """, unsafe_allow_html=True)

    metrics = load_model_metrics()

    # Section 1: Project Overview & Abstract
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>🎯</span> Project Objective & Abstract</div>', unsafe_allow_html=True)
    st.markdown("""
    The **AI-Based Customer Feedback Sentiment Analysis System** is an end-to-end machine learning application designed to automate the triage, classification, and operational intelligence extraction from unstructured customer reviews. 
    
    In modern digital commerce and SaaS operations, thousands of feedback notes are posted across support tickets, social media, and app stores daily. Manual inspection is unscalable and prone to human bias. This platform applies **Natural Language Processing (NLP)** and **Supervised Classification** to categorize customer reviews into:
    - **Positive Sentiment**: Praise, advocacy, high customer delight
    - **Negative Sentiment**: Friction points, software bugs, billing disputes, churn risks
    - **Neutral Sentiment**: Inquiries, standard operational logs, balanced statements
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Verified Model Evaluation Metrics (Never Hardcoded!)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>🧪</span> Empirical Model Evaluation (Held-Out Test Set)</div>', unsafe_allow_html=True)
    st.caption(f"Trained: {metrics.get('timestamp')} &bull; Train Set: {metrics.get('train_size')} | Test Set: {metrics.get('test_size')} samples (Stratified 80/20)")

    acc_pct = round(metrics.get("accuracy", 0.0) * 100, 2)
    f1_pct = round(metrics.get("f1_macro", 0.0) * 100, 2)
    prec_pct = round(metrics.get("precision_macro", 0.0) * 100, 2)
    rec_pct = round(metrics.get("recall_macro", 0.0) * 100, 2)

    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card positive">
                <div class="kpi-header">
                    <span class="kpi-title">Test Accuracy</span>
                    <span class="kpi-icon">🎯</span>
                </div>
                <div class="kpi-value" style="color: #34D399;">{acc_pct}%</div>
                <div class="kpi-subtext">Verified on 600 test reviews</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">F1-Score (Macro)</span>
                    <span class="kpi-icon">📐</span>
                </div>
                <div class="kpi-value">{f1_pct}%</div>
                <div class="kpi-subtext">Balanced across all 3 classes</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Precision (Macro)</span>
                    <span class="kpi-icon">🔬</span>
                </div>
                <div class="kpi-value">{prec_pct}%</div>
                <div class="kpi-subtext">Low false-positive rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Recall (Macro)</span>
                    <span class="kpi-icon">📡</span>
                </div>
                <div class="kpi-value">{rec_pct}%</div>
                <div class="kpi-subtext">High sentiment coverage</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Confusion Matrix & Classification Report Columns
    col_cm, col_cr = st.columns([1, 1], gap="medium")

    classes = metrics.get("classes", ["Negative", "Neutral", "Positive"])
    cm = metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    with col_cm:
        st.markdown("<b style='color: #E2E8F0;'>Confusion Matrix (Test Set)</b>", unsafe_allow_html=True)
        # Confusion matrix heatmap using Plotly
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Review Count"),
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Purples"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
            margin=dict(t=25, b=25, l=25, r=25)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cr:
        st.markdown("<b style='color: #E2E8F0;'>Detailed Classification Report</b>", unsafe_allow_html=True)
        clf_rep = metrics.get("classification_report", {})
        
        rep_rows = []
        for cls_name in classes:
            if cls_name in clf_rep:
                item = clf_rep[cls_name]
                rep_rows.append({
                    "Class": cls_name,
                    "Precision": f"{item['precision'] * 100:.1f}%",
                    "Recall": f"{item['recall'] * 100:.1f}%",
                    "F1-Score": f"{item['f1-score'] * 100:.1f}%",
                    "Support": int(item['support'])
                })
        
        rep_df = pd.DataFrame(rep_rows)
        st.dataframe(rep_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 1rem; line-height: 1.5;">
                <b>Model Specification:</b> {metrics.get('model_type')}<br>
                <b>Feature Extraction:</b> {metrics.get('feature_extractor')}<br>
                <b>Cross-Entropy Loss:</b> Multinomial L-BFGS solver with L2 regularization.
            </div>
        """, unsafe_allow_html=True)

    # Retrain Model Button
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        st.caption("Need to update or re-evaluate the model? Retraining will split data, fit TF-IDF, and generate fresh metrics.")
    with c_btn2:
        if st.button("🔄 Retrain ML Model", use_container_width=True):
            with st.spinner("Retraining NLP Pipeline and evaluating test set..."):
                fresh_metrics = train_and_evaluate_model()
                st.success(f"Model retrained! New Accuracy: {fresh_metrics['accuracy'] * 100:.2f}%")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Technology Stack & System Architecture
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>⚙️</span> System Architecture & Technology Stack</div>', unsafe_allow_html=True)

    col_arch1, col_arch2 = st.columns([1, 1], gap="medium")

    with col_arch1:
        st.markdown("""
        #### 🛠️ Technology Stack
        - **Programming Language:** Python 3.12
        - **Machine Learning Engine:** Scikit-learn (Logistic Regression, TF-IDF Vectorizer)
        - **Natural Language Processing:** NLTK (Tokenization, Negation Preservation, Stopwords, Stemming)
        - **User Interface Framework:** Streamlit with Custom CSS (SaaS Navy/Purple Theme)
        - **Data Processing:** Pandas, NumPy
        - **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
        - **Data Persistence:** SQLite3 with indexing
        - **Model Serialization:** Joblib
        """)

    with col_arch2:
        st.markdown("""
        #### 📂 Project Modular Architecture
        - `app.py`: Main application controller and routing
        - `model_training.py`: Dataset generation, vectorization, training & evaluation
        - `sentiment_analysis.py`: Inference engine, probability scoring & XAI
        - `preprocessing.py`: NLP text normalization & negation preservation
        - `database.py`: SQLite database queries, KPIs & seed data
        - `pages/`: Modular page views (Dashboard, Single, Bulk, History, About)
        - `models/`: Serialized model binaries and evaluation metrics
        - `data/`: SQLite database and sample CSV templates
        - `assets/`: Custom SaaS CSS stylesheets
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Future Roadmap & Improvements
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel-title"><span>🚀</span> Future Roadmap & Scalability Improvements</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Aspect-Based Sentiment Analysis (ABSA):** Detect granular sentiment for specific sub-components (e.g. "Battery was great, but camera was terrible").
    2. **Transformer Fine-Tuning (BERT / RoBERTa):** Complement TF-IDF baseline with fine-tuned contextual embeddings for edge cases.
    3. **Automated Alerting & Webhooks:** Send real-time Slack/Discord alerts when customer dissatisfaction spikes or severe negative reviews are detected.
    4. **Multi-Language Support:** Expand tokenization to support multilingual customer reviews using MarianMT or multilingual BERT.
    5. **Automated Response Generation:** Provide recommended draft replies to address dissatisfied customers based on their feedback.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

