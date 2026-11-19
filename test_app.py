"""
Comprehensive Test Suite for AI-Based Customer Feedback Sentiment Analysis System.
Verifies all modules, pipelines, database queries, and inference routines.
"""

import os
import unittest
import pandas as pd
import numpy as np

from preprocessing import clean_text, expand_contractions, get_text_statistics
from database import (
    init_db,
    insert_feedback,
    insert_bulk_feedback,
    get_all_feedback,
    get_feedback_statistics,
    delete_feedback,
    clear_all_feedback,
    seed_sample_data
)
from sentiment_analysis import (
    predict_sentiment,
    predict_bulk,
    load_model_metrics,
    load_model_and_vectorizer
)


class TestSentimentSystem(unittest.TestCase):

    def test_01_preprocessing(self):
        """Test text cleaning, contraction expansion and negation retention."""
        raw = "I didn't like the product! Visit https://example.com <p>email me at test@example.com</p>"
        cleaned = clean_text(raw, remove_stopwords=True)
        self.assertIn("not", cleaned, "Negation 'not' must be retained during cleaning")
        self.assertNotIn("http", cleaned, "URLs must be stripped")
        self.assertNotIn("<p>", cleaned, "HTML tags must be stripped")
        self.assertNotIn("test@example.com", cleaned, "Emails must be stripped")

        stats = get_text_statistics("Hello world this is a test")
        self.assertEqual(stats["word_count"], 6)

    def test_02_model_loading_and_inference(self):
        """Test model loading, classification and probability scores."""
        model, vec = load_model_and_vectorizer()
        self.assertIsNotNone(model)
        self.assertIsNotNone(vec)

        # Positive test
        pos_res = predict_sentiment("The product exceeded all my expectations! Highly recommend.")
        self.assertEqual(pos_res["sentiment"], "Positive")
        self.assertGreater(pos_res["confidence"], 0.5)
        self.assertIn("Positive", pos_res["probabilities"])

        # Negative test
        neg_res = predict_sentiment("Terrible quality. Broke after two days and customer support was rude.")
        self.assertEqual(neg_res["sentiment"], "Negative")
        self.assertGreater(neg_res["confidence"], 0.5)

        # Neutral test
        neu_res = predict_sentiment("The package arrived on Tuesday as per the delivery schedule.")
        self.assertEqual(neu_res["sentiment"], "Neutral")

    def test_03_explainability_tokens(self):
        """Test XAI token attribution."""
        res = predict_sentiment("The build quality is excellent and superb.")
        self.assertIn("token_explanations", res)
        tokens = [t["word"] for t in res["token_explanations"]]
        self.assertTrue(any(w in tokens for w in ["excellent", "superb", "quality"]))

    def test_04_bulk_inference(self):
        """Test vectorized batch processing on a DataFrame."""
        test_df = pd.DataFrame({
            "id": ["C1", "C2", "C3"],
            "review": [
                "Awesome service and friendly staff!",
                "Worst experience ever, complete fraud.",
                "Order received on schedule."
            ]
        })
        analyzed = predict_bulk(test_df, text_column="review")
        self.assertEqual(len(analyzed), 3)
        self.assertIn("Predicted_Sentiment", analyzed.columns)
        self.assertIn("Confidence", analyzed.columns)
        self.assertEqual(analyzed.iloc[0]["Predicted_Sentiment"], "Positive")
        self.assertEqual(analyzed.iloc[1]["Predicted_Sentiment"], "Negative")
        self.assertEqual(analyzed.iloc[2]["Predicted_Sentiment"], "Neutral")

    def test_05_database_crud(self):
        """Test SQLite persistence, filtering, statistics, and deletion."""
        test_db = os.path.join(os.path.dirname(__file__), "data", "test_feedback.db")
        if os.path.exists(test_db):
            os.remove(test_db)

        init_db(test_db)

        # Insert single
        row_id = insert_feedback(
            review_text="Great experience with customer care!",
            sentiment="Positive",
            confidence=0.95,
            cleaned_text="great experience customer care",
            category="Customer Service",
            source="single",
            customer_id="CUST-TEST-1",
            db_path=test_db
        )
        self.assertGreater(row_id, 0)

        # Insert bulk
        bulk_rows = [
            {"review_text": "Bad battery", "cleaned_text": "bad battery", "sentiment": "Negative", "confidence": 0.90, "category": "Product Quality", "source": "bulk", "customer_id": "CUST-TEST-2"},
            {"review_text": "Standard item", "cleaned_text": "standard item", "sentiment": "Neutral", "confidence": 0.85, "category": "General", "source": "bulk", "customer_id": "CUST-TEST-3"}
        ]
        count = insert_bulk_feedback(bulk_rows, db_path=test_db)
        self.assertEqual(count, 2)

        # Query all
        all_df = get_all_feedback(db_path=test_db)
        self.assertEqual(len(all_df), 3)

        # Query filtered
        pos_df = get_all_feedback(sentiment_filter="Positive", db_path=test_db)
        self.assertEqual(len(pos_df), 1)

        # Statistics
        stats = get_feedback_statistics(db_path=test_db)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["positive"], 1)
        self.assertEqual(stats["negative"], 1)
        self.assertEqual(stats["neutral"], 1)

        # Delete
        deleted = delete_feedback(row_id, db_path=test_db)
        self.assertTrue(deleted)
        stats_after = get_feedback_statistics(db_path=test_db)
        self.assertEqual(stats_after["total"], 2)

        # Clean up test db
        if os.path.exists(test_db):
            os.remove(test_db)

    def test_06_model_metrics(self):
        """Test that model metrics are present and non-zero."""
        metrics = load_model_metrics()
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.8)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("classification_report", metrics)


if __name__ == "__main__":
    unittest.main()

