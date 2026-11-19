# Sentify AI — AI-Based Customer Feedback Sentiment Analysis System

> **College Internship Capstone Project** | 8-Week Software Development Internship  
> **Tech Stack:** Python 3.12 • Scikit-learn • NLTK • Streamlit • Plotly • SQLite

---

## 🎯 Project Overview

**Sentify AI** is a full-stack, production-quality AI web application that automatically classifies customer reviews and feedback into three sentiment categories:

| Sentiment | Description |
|-----------|-------------|
| ✅ **Positive** | Praise, advocacy, high satisfaction |
| ❌ **Negative** | Complaints, bugs, billing disputes, churn risk |
| ⚖️ **Neutral** | Factual statements, standard operations, inquiries |

The system uses a **TF-IDF + Logistic Regression** NLP pipeline trained on a curated 3,000-sample dataset, wrapped inside a **professional dark-navy/purple SaaS dashboard** built with Streamlit.

---

## 🏗️ Project Architecture

```
sentiment-analysis-system/
│
├── app.py                      # Main Streamlit entry point & navigation controller
├── model_training.py           # Dataset generation, TF-IDF, training & evaluation
├── sentiment_analysis.py       # Inference engine, probability scoring, XAI tokens
├── preprocessing.py            # NLP text cleaning, negation preservation
├── database.py                 # SQLite CRUD, seed data, statistics queries
├── test_app.py                 # Comprehensive automated test suite (6 tests)
├── requirements.txt            # Python dependency list
│
├── pages/
│   ├── __init__.py
│   ├── dashboard_page.py       # Executive KPI dashboard with live charts
│   ├── single_analysis_page.py # Interactive single-review analyzer + XAI
│   ├── bulk_analysis_page.py   # CSV upload, batch inference & download
│   ├── history_page.py         # SQLite feedback archive with filters
│   └── about_page.py           # ML metrics, confusion matrix, project info
│
├── data/
│   ├── customer_reviews_dataset.csv   # 3,000-sample labeled training dataset
│   ├── sample_customer_reviews.csv    # 25-review ready-to-test bulk upload file
│   └── feedback.db                    # SQLite database (auto-created)
│
├── models/
│   ├── sentiment_model.joblib         # Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF vectorizer
│   └── model_metrics.json            # Verified evaluation metrics (never hardcoded)
│
└── assets/
    └── style.css                      # Custom SaaS dark navy/purple CSS theme
```

---

## ⚙️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Machine Learning | Scikit-learn (Logistic Regression) |
| Feature Extraction | TF-IDF Vectorizer (1–2 n-grams, 5,000 features) |
| NLP Preprocessing | NLTK (stopwords, stemming, negation preservation) |
| Frontend | Streamlit 1.67+ with custom CSS |
| Visualizations | Plotly Express & Plotly Graph Objects |
| Database | SQLite3 (via Python `sqlite3` stdlib) |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 🚀 Installation & Setup

### Step 1 — Clone or navigate to the project directory

```bash
cd /path/to/sentiment-analysis-system
```

### Step 2 — Create and activate a Python 3.12 virtual environment

```bash
python3.12 -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download NLTK data (one-time only)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

### Step 5 — Train the ML model (runs automatically on first launch, but can be run manually)

```bash
python model_training.py
```

### Step 6 — Run the Streamlit application

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📱 Application Pages

### 1. 📊 Dashboard
- **5 Live KPI Cards:** Total reviews, Positive, Negative, Neutral counts, CSAT%
- **Donut Chart:** Sentiment distribution with interactive hover
- **Grouped Bar Chart:** Category vs. Sentiment breakdown
- **Line Chart:** Sentiment volume trend over time
- **Recent Feed:** Latest reviews with badges and confidence scores

### 2. 💬 Sentiment Analysis (Single Review)
- Text area with 4 quick-load preset samples (Positive, Negative, Neutral, Negation test)
- Category & Customer ID fields
- **Result Card:** Large emoji, sentiment name, color-coded confidence badge
- **Probability Progress Bars:** Per-class confidence scores
- **XAI Token Panel:** Most impactful words with positive/negative scores
- Auto-save to SQLite database (toggleable)

### 3. 📁 Bulk Analysis
- CSV file uploader with smart column detection
- Batch TF-IDF vectorization for high-speed inference
- Summary KPI cards for the uploaded batch
- Pie chart + Confidence histogram
- Filterable data table with all results
- **Download Analyzed CSV** and **Save to Database** buttons

### 4. 📜 Feedback History
- Full-text search and sentiment/category/source filters
- Paginated sortable data table with SQLite records
- Export to CSV, delete individual records, re-seed or clear database
- Top frequent vocabulary words panel per sentiment class

### 5. ℹ️ About Project
- Verified ML metrics: Accuracy, Precision, Recall, F1 (from actual test set)
- Confusion matrix heatmap + per-class classification report table
- **Retrain Model** button that regenerates real metrics
- System architecture, tech stack, and future roadmap

---

## 🧪 Running Tests

```bash
python test_app.py
```

All 6 automated tests cover:
1. ✅ NLP text preprocessing and negation preservation
2. ✅ Model loading and 3-class inference
3. ✅ Explainable AI (XAI) token attribution
4. ✅ Bulk batch DataFrame inference
5. ✅ SQLite CRUD: insert, query, filter, stats, delete
6. ✅ Model metrics validation (accuracy > 80%)

---

## 🤖 Machine Learning Pipeline

```
Raw Text Input
     │
     ▼
Preprocessing (preprocessing.py)
  - Unicode normalization
  - Contraction expansion (don't → do not)
  - URL / HTML / Email removal
  - Lowercase + tokenization
  - Negation-preserving stopword filtering
     │
     ▼
TF-IDF Vectorization
  - Unigrams + Bigrams (n-gram range: 1–2)
  - 5,000 max features
  - Sublinear TF scaling
     │
     ▼
Logistic Regression Classifier
  - Solver: L-BFGS
  - C = 1.5 (L2 regularization)
  - class_weight = 'balanced'
  - max_iter = 1,000
     │
     ▼
Prediction + Probabilities + XAI Tokens
```

**Training Dataset:**  
- 3,000 labeled customer reviews (1,000 per class)  
- 6 topic domains: Product Quality, Shipping, Customer Service, Pricing, Software/UX, General  
- Stratified 80/20 train/test split  

---

## 📈 Model Performance (from Actual Test Set)

| Metric | Score |
|--------|-------|
| Test Accuracy | Computed at runtime from held-out test set |
| Precision (Macro) | Computed at runtime |
| Recall (Macro) | Computed at runtime |
| F1-Score (Macro) | Computed at runtime |

> **Note:** Metrics are never hardcoded. Every value displayed in the About page is loaded from `models/model_metrics.json`, which is generated by running the actual evaluation on the held-out test set. Use the **Retrain Model** button to regenerate fresh metrics.

---

## 💡 Key Design Decisions

1. **Negation Preservation:** Standard stopword lists remove words like "not", "no", "never" — which completely destroys sentiment. This system retains all negation words to preserve meaning (e.g. "not bad" vs "bad").

2. **Contraction Expansion:** Handles 50+ English contractions (wasn't → was not) before vectorization so negations survive tokenization.

3. **Balanced Classes:** `class_weight='balanced'` in LogisticRegression corrects for any class imbalance in production data.

4. **XAI Transparency:** Sentiment decisions are explained by showing the top model coefficient weights for the predicted class — giving users insight into *why* a decision was made.

5. **Real Metrics:** Model accuracy is computed on a proper stratified held-out test split and stored in JSON — never invented or hardcoded.

---

## 🔭 Future Improvement Roadmap

1. **Aspect-Based Sentiment (ABSA):** Identify sentiment per product dimension (battery, camera, shipping)
2. **Transformer Models:** Fine-tune BERT/RoBERTa for context-aware edge cases ("not bad", "could be worse")
3. **Real-Time Alerting:** Slack/Discord webhooks when negative spikes detected
4. **Multi-Language Support:** Extend to French, Spanish, German reviews via multilingual BERT
5. **Automated Response Drafts:** Generate reply suggestions for negative reviews using LLM integration
6. **REST API Backend:** Expose `/predict` endpoint (FastAPI) for external integration

---

## 📚 Sample Customer Reviews for Demo

| Review | Expected Sentiment |
|--------|-------------------|
| "The product quality is absolutely outstanding and exceeded all expectations." | Positive |
| "Worst customer service ever! Waited on hold and was hung up on." | Negative |
| "The order was confirmed via email and tracking link was provided." | Neutral |
| "Five stars! Polite, professional, and resolved my issue in minutes." | Positive |
| "Terrible build quality. Broke after two days. Requesting a refund." | Negative |

---

## 👨‍💻 Project Information

- **Project Title:** AI-Based Customer Feedback Sentiment Analysis System  
- **Internship Duration:** 8 Weeks  
- **Stack:** Python, Scikit-learn, NLTK, Streamlit, Plotly, SQLite  
- **Model Type:** TF-IDF + Logistic Regression (Multi-class)  
- **Classes:** Positive, Negative, Neutral  

---

*Built with ❤️ as a college internship capstone project. All code is modular, beginner-friendly, and thoroughly commented.*

