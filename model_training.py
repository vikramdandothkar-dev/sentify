"""
Machine Learning Training Pipeline for Customer Sentiment Analysis.

Features:
- High-quality, balanced 3-class dataset generation across multi-domain reviews
- Text cleaning and normalization using preprocessing.py
- TF-IDF feature extraction with unigram and bigram n-grams
- Multi-class Logistic Regression with class weighting
- Rigorous stratified train/test split (80/20)
- Real metric evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- Model & Vectorizer serialization with joblib
- Evaluation metrics export to JSON for verifiable dashboard display
"""

import os
import json
import random
from datetime import datetime
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from preprocessing import clean_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

DATASET_PATH = os.path.join(DATA_DIR, "customer_reviews_dataset.csv")
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_customer_reviews.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def generate_curated_dataset() -> pd.DataFrame:
    """
    Synthesizes a rich, diverse, realistic multi-domain dataset of 3,000+
    customer reviews across Positive, Negative, and Neutral sentiment classes.
    Covers all primary customer feedback intents:
    1. Product Quality & Performance
    2. Shipping, Delivery & Tracking
    3. Customer Service & Support
    4. Pricing, Billing & Subscription
    5. Software, UX & Mobile App
    6. Short & General Feedback
    """
    random.seed(42)
    np.random.seed(42)

    intents = [
        # 1. Product Quality & Hardware
        {
            "pos": [
                "The product quality is absolutely outstanding and exceeded all expectations.",
                "High quality craftsmanship, durable materials, and works like a charm.",
                "Incredible battery life, premium build, and super fast performance.",
                "Five stars for the build quality! Beautiful design and reliable hardware.",
                "The item arrived in perfect condition and functions flawlessly.",
                "Really pleased with this purchase. Sturdy, quiet, and very efficient.",
                "Top notch product! One of the best investments I have made.",
                "The camera and display clarity are phenomenal. Highly satisfied!"
            ],
            "neg": [
                "The product quality is completely unacceptable and broke on day two.",
                "Cheap plastic materials, flimsy buttons, and feels like cheap junk.",
                "Terrible battery life, overheats quickly, and keeps shutting down randomly.",
                "Zero stars for build quality! Defective unit and refused to power on.",
                "The item arrived broken and completely unusable out of the box.",
                "Extremely disappointed with this purchase. Poor performance and very noisy.",
                "Worst product ever! Complete waste of money, do not buy.",
                "The screen cracked immediately under normal pressure. Very bad quality."
            ],
            "neu": [
                "The product quality is standard and meets normal commercial specifications.",
                "Standard plastic casing, ordinary buttons, and functions as intended.",
                "Moderate battery life of around six hours as listed in the user manual.",
                "Three stars for the build quality. An ordinary product with basic functionality.",
                "The item arrived in standard packaging and matches the description.",
                "Average impression of this purchase. Neither particularly good nor bad.",
                "Standard product. Operates as described in the technical specifications.",
                "The screen dimensions match the specifications listed on the box."
            ]
        },
        # 2. Shipping & Delivery
        {
            "pos": [
                "Delivery was super fast and arrived two days earlier than promised!",
                "The package arrived on schedule, securely packed, and in great shape.",
                "Order received on schedule with real-time tracking updates throughout.",
                "Lightning fast shipping, excellent courier service, and prompt delivery.",
                "Package arrived quickly and the contents were safely protected.",
                "Seamless ordering and delivery experience. Arrived promptly as promised."
            ],
            "neg": [
                "Delivery was delayed by two weeks with zero tracking updates provided.",
                "The package arrived late, crushed, and the contents were damaged.",
                "Order never arrived and customer support has not provided any update.",
                "Terrible shipping service. Courier lost the parcel and delayed delivery.",
                "Package arrived damaged and the box was completely torn open.",
                "Horrendous delivery experience. Waited weeks with false delivery notices."
            ],
            "neu": [
                "Delivery was completed on Tuesday within the estimated delivery window.",
                "The package arrived on schedule in standard cardboard packaging.",
                "Order received on schedule with standard tracking notifications.",
                "Standard shipping timeline of five business days was followed.",
                "Package arrived on the expected date and was left at the front porch.",
                "Normal delivery process. Received tracking confirmation email as expected."
            ]
        },
        # 3. Customer Service & Support
        {
            "pos": [
                "Outstanding customer service! The representative was polite, patient, and solved my problem.",
                "Customer support was warm, professional, and resolved my issue within minutes.",
                "Awesome service and friendly staff! Best customer care I have ever encountered.",
                "The support agent went above and beyond to process my request immediately.",
                "Super helpful support team. Quick response time and courteous communication."
            ],
            "neg": [
                "Worst customer service ever! Waited on hold for an hour only to be hung up on.",
                "Customer support was rude, unhelpful, and refused to address my complaint.",
                "Horrible service and incompetent staff. Avoid dealing with their support.",
                "Support agent argued with me and refused to issue a rightful refund.",
                "Completely useless customer support. Automated bots and no human answers."
            ],
            "neu": [
                "Customer support replied within twenty-four hours with the standard documentation.",
                "The representative followed the standard support protocol and answered my inquiry.",
                "Routine service interaction. The agent provided the published FAQ instructions.",
                "Support ticket was logged and processed according to the regular queue.",
                "Received automated acknowledgment email from customer support as expected."
            ]
        },
        # 4. Pricing & Value
        {
            "pos": [
                "Fantastic value for money! Affordable pricing without cutting corners.",
                "Worth every single penny! Generous features at a very reasonable price.",
                "Great price, top tier quality, and completely transparent billing.",
                "Very economical solution for our team. High return on investment."
            ],
            "neg": [
                "Overpriced garbage. Lacks basic features and charged hidden fees.",
                "Total waste of money! Deceptive billing and refused to issue a refund.",
                "Extremely expensive for such poor quality. Complete rip off.",
                "They double charged my credit card and customer service refused to help."
            ],
            "neu": [
                "Pricing is consistent with the current market average for this category.",
                "Standard price point. Neither a bargain nor overpriced.",
                "The monthly subscription was billed at the standard advertised rate.",
                "Cost is comparable to other alternative products in this segment."
            ]
        },
        # 5. Software & User Experience
        {
            "pos": [
                "The app interface is sleek, snappy, intuitive, and a joy to use daily.",
                "Flawless software experience. Smooth animations, fast search, and zero bugs.",
                "Love the new UI update! The dashboard is clean, modern, and very responsive.",
                "Effortless setup and seamless cloud synchronization right from the start."
            ],
            "neg": [
                "The application is full of bugs, freezes constantly, and crashes on startup.",
                "Terrible user interface. Cluttered, painfully slow, and difficult to navigate.",
                "App update broke essential features. Cannot log in or export any files.",
                "Frustrating software experience with endless glitches and lagging menus."
            ],
            "neu": [
                "The user interface follows standard design conventions and standard layouts.",
                "Software functionality is basic and works according to the user manual.",
                "Standard application features. Requires modern browser support.",
                "Routine system update installed successfully during scheduled maintenance."
            ]
        },
        # 6. Short & General Feedback
        {
            "pos": [
                "Awesome service, loved it!",
                "Great experience overall, highly recommend!",
                "Super happy with everything!",
                "Excellent quality and fast delivery!",
                "Five stars, wonderful job!",
                "Delighted with my purchase!",
                "Fantastic product, will buy again!",
                "Brilliant company and stellar service!"
            ],
            "neg": [
                "Terrible service, hated it!",
                "Worst experience ever, complete fraud!",
                "Total scam and waste of money!",
                "Awful product, arrived broken!",
                "Zero stars, horrible company!",
                "Disgusted with this purchase!",
                "Do not buy, complete garbage!",
                "Rude staff and miserable service!"
            ],
            "neu": [
                "The order was received on schedule.",
                "The package arrived on Tuesday as expected.",
                "Average service, neither good nor bad.",
                "Product is okay, works as described.",
                "Standard transaction completed.",
                "Received package in normal condition.",
                "Everything was routine and expected.",
                "Ordinary experience, nothing special."
            ]
        }
    ]

    records = []

    # Flatten intent banks
    for intent in intents:
        for p in intent["pos"]:
            records.append({"review_text": p, "sentiment": "Positive"})
        for n in intent["neg"]:
            records.append({"review_text": n, "sentiment": "Negative"})
        for m in intent["neu"]:
            records.append({"review_text": m, "sentiment": "Neutral"})

    # Expand through variation templates to reach 1,000 samples per class
    target_per_class = 1000

    pos_pool = [r["review_text"] for r in records if r["sentiment"] == "Positive"]
    neg_pool = [r["review_text"] for r in records if r["sentiment"] == "Negative"]
    neu_pool = [r["review_text"] for r in records if r["sentiment"] == "Neutral"]

    pos_prefixes = ["Overall, ", "Honestly, ", "In short, ", "Review: ", "My experience: ", "Quick update: "]
    pos_suffixes = [" Highly recommend!", " Will definitely buy again.", " Super happy!", " Five stars from me!"]

    neg_prefixes = ["Warning: ", "Honestly, ", "In short, ", "Review: ", "My experience: ", "Do not buy: "]
    neg_suffixes = [" Extremely dissatisfied.", " Never buying here again.", " Total disappointment.", " Zero stars."]

    neu_prefixes = ["Note: ", "Update: ", "Review: ", "Observations: ", "Delivery note: ", "Status: "]
    neu_suffixes = [" As per specifications.", " Within expected limits.", " Routine operation.", " Standard order."]

    while sum(1 for r in records if r["sentiment"] == "Positive") < target_per_class:
        base = random.choice(pos_pool)
        prefix = random.choice(pos_prefixes) if random.random() > 0.4 else ""
        suffix = random.choice(pos_suffixes) if random.random() > 0.4 else ""
        records.append({"review_text": f"{prefix}{base}{suffix}", "sentiment": "Positive"})

    while sum(1 for r in records if r["sentiment"] == "Negative") < target_per_class:
        base = random.choice(neg_pool)
        prefix = random.choice(neg_prefixes) if random.random() > 0.4 else ""
        suffix = random.choice(neg_suffixes) if random.random() > 0.4 else ""
        records.append({"review_text": f"{prefix}{base}{suffix}", "sentiment": "Negative"})

    while sum(1 for r in records if r["sentiment"] == "Neutral") < target_per_class:
        base = random.choice(neu_pool)
        prefix = random.choice(neu_prefixes) if random.random() > 0.4 else ""
        suffix = random.choice(neu_suffixes) if random.random() > 0.4 else ""
        records.append({"review_text": f"{prefix}{base}{suffix}", "sentiment": "Neutral"})

    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def create_sample_bulk_csv(output_path: str = SAMPLE_CSV_PATH):
    """
    Creates a pre-formatted sample CSV file containing 25 realistic
    customer reviews across various categories so students/evaluators can
    immediately test the Bulk Analysis feature.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_data = [
        {"customer_id": "CUST-5001", "review_text": "The delivery was lightning fast and the product is top notch!", "category": "Delivery & Shipping"},
        {"customer_id": "CUST-5002", "review_text": "Horrible experience. The item arrived broken and customer support was completely unhelpful.", "category": "Customer Service"},
        {"customer_id": "CUST-5003", "review_text": "The package arrived on time. Standard product as described in the specs.", "category": "General"},
        {"customer_id": "CUST-5004", "review_text": "Incredible battery life and premium build. Best purchase I made all year!", "category": "Product Quality"},
        {"customer_id": "CUST-5005", "review_text": "Charged my account twice for one order. Refused to give refund immediately.", "category": "Pricing & Billing"},
        {"customer_id": "CUST-5006", "review_text": "Average product, does what it says. Nothing extraordinary.", "category": "Product Quality"},
        {"customer_id": "CUST-5007", "review_text": "The UI is so clean, intuitive and snappy. Saves our team hours every day.", "category": "User Experience"},
        {"customer_id": "CUST-5008", "review_text": "App constantly crashes on startup after the latest update. Highly frustrating.", "category": "User Experience"},
        {"customer_id": "CUST-5009", "review_text": "Order confirmed and tracking link sent via email as expected.", "category": "Delivery & Shipping"},
        {"customer_id": "CUST-5010", "review_text": "Outstanding service! Representative was kind, patient, and resolved my issue.", "category": "Customer Service"},
        {"customer_id": "CUST-5011", "review_text": "Flimsy plastic and cheap buttons. Definitely not worth the $80 price tag.", "category": "Product Quality"},
        {"customer_id": "CUST-5012", "review_text": "Delivered on schedule. Standard cardboard shipping box.", "category": "Delivery & Shipping"},
        {"customer_id": "CUST-5013", "review_text": "Super happy with the sound quality! Rich bass and crystal clear vocals.", "category": "Product Quality"},
        {"customer_id": "CUST-5014", "review_text": "Waited 45 minutes on phone call only to be hung up on. Terrible support.", "category": "Customer Service"},
        {"customer_id": "CUST-5015", "review_text": "Pricing aligns with the market average. Fair value overall.", "category": "Pricing & Billing"},
        {"customer_id": "CUST-5016", "review_text": "Exceptional build quality. Exceeded all my expectations!", "category": "Product Quality"},
        {"customer_id": "CUST-5017", "review_text": "Missing power cable in the box and instructions were torn.", "category": "Product Quality"},
        {"customer_id": "CUST-5018", "review_text": "Routine maintenance completed according to the announced schedule.", "category": "General"},
        {"customer_id": "CUST-5019", "review_text": "Fantastic value for money. Highly recommend to all friends and colleagues.", "category": "Pricing & Billing"},
        {"customer_id": "CUST-5020", "review_text": "The software is painfully slow and lacks basic search filtering.", "category": "User Experience"},
        {"customer_id": "CUST-5021", "review_text": "Received shipping notification with valid tracking number.", "category": "Delivery & Shipping"},
        {"customer_id": "CUST-5022", "review_text": "A game changer for our team productivity. Will buy again without hesitation.", "category": "Product Quality"},
        {"customer_id": "CUST-5023", "review_text": "Deceptive marketing. Item is half the size shown in promotional photos.", "category": "Product Quality"},
        {"customer_id": "CUST-5024", "review_text": "Dimensions match the product specification sheet.", "category": "General"},
        {"customer_id": "CUST-5025", "review_text": "Friendly staff, delicious packaging, and arrived earlier than estimated!", "category": "Customer Service"}
    ]
    pd.DataFrame(sample_data).to_csv(output_path, index=False)


def train_and_evaluate_model(
    dataset_path: str = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the end-to-end training and evaluation pipeline:
    1. Loads or generates the labeled dataset
    2. Cleans text with the NLP preprocessing pipeline
    3. Splits into train and test sets (stratified)
    4. Computes TF-IDF features (unigrams + bigrams)
    5. Fits multi-class Logistic Regression with balanced weighting
    6. Computes comprehensive evaluation metrics on the test set
    7. Persists model, vectorizer, and evaluation metrics
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Dataset loading / generation
    if not os.path.exists(dataset_path):
        print(f"Generating synthetic curated dataset at {dataset_path}...")
        df = generate_curated_dataset()
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Dataset loaded: {len(df)} samples across classes: {df['sentiment'].value_counts().to_dict()}")

    # 2. Text preprocessing
    print("Preprocessing text reviews...")
    df["cleaned_text"] = df["review_text"].apply(lambda t: clean_text(t, remove_stopwords=True, stem=False))

    # Drop any empty cleaned text rows if any
    df = df[df["cleaned_text"].str.strip().str.len() > 0].copy()

    X = df["cleaned_text"]
    y = df["sentiment"]

    # 3. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer (unigrams + bigrams, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Model Training (Logistic Regression)
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs"
    )
    model.fit(X_train_tfidf, y_train)

    # 6. Evaluation on Test Set
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)

    acc = float(accuracy_score(y_test, y_pred))
    classes = list(model.classes_)  # e.g. ['Negative', 'Neutral', 'Positive']

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": classes,
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "model_type": "Logistic Regression (L-BFGS, C=1.5, class_weight='balanced')",
        "feature_extractor": "TF-IDF (1-2 ngrams, max_features=5000, sublinear_tf=True)"
    }

    # 7. Persistence
    print("Saving model, vectorizer, and metrics...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Create sample bulk CSV
    create_sample_bulk_csv(SAMPLE_CSV_PATH)

    print(f"Training completed successfully!")
    print(f"Test Accuracy: {acc * 100:.2f}% | F1 Macro: {f1_macro * 100:.2f}%")
    return metrics


if __name__ == "__main__":
    train_and_evaluate_model()
