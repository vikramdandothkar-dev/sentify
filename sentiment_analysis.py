"""
Sentiment Analysis Inference & Explainability Module.

Provides:
- Model loading and caching
- Single-text prediction with probability distribution
- Explainable AI (XAI) token contribution analysis using TF-IDF weights and Logistic Regression coefficients
- High-performance bulk/batch inference for DataFrames
- Evaluation metrics loader
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from preprocessing import clean_text, get_text_statistics

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Module-level cached instances
_model = None
_vectorizer = None


def load_model_and_vectorizer():
    """
    Loads and caches the trained Logistic Regression model and TF-IDF vectorizer.
    If models do not exist, runs the training pipeline automatically.
    """
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        from model_training import train_and_evaluate_model
        train_and_evaluate_model()

    _model = joblib.load(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def get_token_explanations(
    cleaned_text: str,
    predicted_sentiment: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Explainable AI (XAI):
    Calculates the individual token impact on the predicted class
    using the TF-IDF feature weights multiplied by Logistic Regression coefficients.
    """
    model, vectorizer = load_model_and_vectorizer()
    if not cleaned_text:
        return []

    tokens = cleaned_text.split()
    feature_names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_
    
    # Locate the target class index
    class_idx = list(model.classes_).index(predicted_sentiment)
    coefs = model.coef_[class_idx]

    token_scores = []
    seen = set()
    for word in tokens:
        if word in vocab and word not in seen:
            seen.add(word)
            idx = vocab[word]
            score = float(coefs[idx])
            token_scores.append({
                "word": word,
                "importance": round(score, 3),
                "impact": "Supports" if score > 0 else "Opposes"
            })

    # Sort by absolute importance
    token_scores.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return token_scores[:top_k]


def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Performs full sentiment classification and probability inference for a single review.
    
    Returns:
        dict: Detailed result containing predicted sentiment, confidence percentage,
              per-class probabilities, text statistics, and token explanations.
    """
    if not text or not str(text).strip():
        return {
            "error": "Empty text provided. Please enter valid customer feedback."
        }

    model, vectorizer = load_model_and_vectorizer()

    # Preprocess text
    cleaned = clean_text(text, remove_stopwords=True)
    if not cleaned.strip():
        # Fallback if text consists only of symbols/stopwords
        cleaned = text.strip().lower()

    # Vectorize
    vector = vectorizer.transform([cleaned])

    # Predict
    predicted_class = str(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    classes = list(model.classes_)

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(classes, probabilities)
    }

    confidence = prob_dict.get(predicted_class, float(np.max(probabilities)))
    explanations = get_token_explanations(cleaned, predicted_class)
    stats = get_text_statistics(text)

    # Style mapping
    style_meta = {
        "Positive": {
            "icon": "😊",
            "color": "#10B981",       # Emerald green
            "bg_color": "rgba(16, 185, 129, 0.12)",
            "border_color": "#10B981",
            "badge_class": "badge-positive",
            "label_description": "Customer is highly satisfied and expressing approval."
        },
        "Negative": {
            "icon": "😡",
            "color": "#EF4444",       # Crimson red
            "bg_color": "rgba(239, 68, 68, 0.12)",
            "border_color": "#EF4444",
            "badge_class": "badge-negative",
            "label_description": "Customer is dissatisfied, reporting an issue or expressing frustration."
        },
        "Neutral": {
            "icon": "😐",
            "color": "#8B5CF6",       # Electric purple / indigo
            "bg_color": "rgba(139, 92, 246, 0.12)",
            "border_color": "#8B5CF6",
            "badge_class": "badge-neutral",
            "label_description": "Factual statement, balanced feedback, or standard operational query."
        }
    }

    meta = style_meta.get(predicted_class, style_meta["Neutral"])

    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "sentiment": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 1),
        "probabilities": prob_dict,
        "prob_positive": prob_dict.get("Positive", 0.0),
        "prob_negative": prob_dict.get("Negative", 0.0),
        "prob_neutral": prob_dict.get("Neutral", 0.0),
        "token_explanations": explanations,
        "text_statistics": stats,
        "style": meta
    }


def predict_bulk(
    df: pd.DataFrame,
    text_column: str,
    id_column: Optional[str] = None,
    category_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Executes vectorized batch inference over a Pandas DataFrame.
    Optimized for high-speed bulk CSV processing.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    model, vectorizer = load_model_and_vectorizer()
    classes = list(model.classes_)

    # Copy and clean text
    result_df = df.copy()
    cleaned_series = result_df[text_column].astype(str).apply(
        lambda t: clean_text(t, remove_stopwords=True)
    )
    result_df["Cleaned_Text"] = cleaned_series

    # Handle any completely empty cleaned strings by replacing with original lowercase
    mask_empty = result_df["Cleaned_Text"].str.strip() == ""
    if mask_empty.any():
        result_df.loc[mask_empty, "Cleaned_Text"] = result_df.loc[mask_empty, text_column].astype(str).str.lower()

    # Batch transform and predict
    features = vectorizer.transform(result_df["Cleaned_Text"])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    result_df["Predicted_Sentiment"] = predictions
    result_df["Confidence"] = np.max(probabilities, axis=1).round(4)
    result_df["Confidence_%"] = (result_df["Confidence"] * 100).round(1)

    for i, cls in enumerate(classes):
        result_df[f"Prob_{cls}"] = probabilities[:, i].round(4)

    return result_df


def load_model_metrics() -> Dict[str, Any]:
    """Loads saved model evaluation metrics and metadata."""
    if not os.path.exists(METRICS_PATH):
        from model_training import train_and_evaluate_model
        return train_and_evaluate_model()

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

