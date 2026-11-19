"""
Database Management Module using SQLite.

Handles all persistence for customer reviews, sentiment predictions,
confidence scores, review categories, timestamps, and analytics metrics.
"""

import os
import sqlite3
from datetime import datetime, timedelta
import random
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

# Default DB Path in the data/ directory
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "feedback.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a database connection with dictionary-like row access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """Initializes the database schema if not already present."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                cleaned_text TEXT,
                sentiment TEXT NOT NULL CHECK(sentiment IN ('Positive', 'Negative', 'Neutral')),
                confidence REAL NOT NULL,
                prob_positive REAL DEFAULT 0.0,
                prob_negative REAL DEFAULT 0.0,
                prob_neutral REAL DEFAULT 0.0,
                category TEXT DEFAULT 'General',
                source TEXT DEFAULT 'single' CHECK(source IN ('single', 'bulk', 'sample')),
                customer_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON feedback(sentiment);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON feedback(created_at);")
        conn.commit()


def insert_feedback(
    review_text: str,
    sentiment: str,
    confidence: float,
    cleaned_text: str = "",
    prob_pos: float = 0.0,
    prob_neg: float = 0.0,
    prob_neu: float = 0.0,
    category: str = "General",
    source: str = "single",
    customer_id: Optional[str] = None,
    created_at: Optional[str] = None,
    db_path: str = DB_PATH
) -> int:
    """Inserts a single analyzed review into the database."""
    init_db(db_path)
    if not customer_id:
        customer_id = f"CUST-{random.randint(1000, 9999)}"
        
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if created_at:
            cursor.execute("""
                INSERT INTO feedback (
                    review_text, cleaned_text, sentiment, confidence,
                    prob_positive, prob_negative, prob_neutral,
                    category, source, customer_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                review_text, cleaned_text, sentiment, float(confidence),
                float(prob_pos), float(prob_neg), float(prob_neu),
                category, source, customer_id, created_at
            ))
        else:
            cursor.execute("""
                INSERT INTO feedback (
                    review_text, cleaned_text, sentiment, confidence,
                    prob_positive, prob_negative, prob_neutral,
                    category, source, customer_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                review_text, cleaned_text, sentiment, float(confidence),
                float(prob_pos), float(prob_neg), float(prob_neu),
                category, source, customer_id
            ))
        conn.commit()
        return cursor.lastrowid


def insert_bulk_feedback(records: List[Dict[str, Any]], db_path: str = DB_PATH) -> int:
    """Inserts multiple analyzed feedback records efficiently."""
    init_db(db_path)
    if not records:
        return 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        rows = []
        for r in records:
            cust_id = r.get("customer_id") or f"CUST-{random.randint(1000, 9999)}"
            rows.append((
                r.get("review_text", ""),
                r.get("cleaned_text", ""),
                r.get("sentiment", "Neutral"),
                float(r.get("confidence", 0.0)),
                float(r.get("prob_positive", 0.0)),
                float(r.get("prob_negative", 0.0)),
                float(r.get("prob_neutral", 0.0)),
                r.get("category", "General"),
                r.get("source", "bulk"),
                cust_id,
                r.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

        cursor.executemany("""
            INSERT INTO feedback (
                review_text, cleaned_text, sentiment, confidence,
                prob_positive, prob_negative, prob_neutral,
                category, source, customer_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        return len(rows)


def get_all_feedback(
    sentiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    search_keyword: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: str = DB_PATH
) -> pd.DataFrame:
    """
    Retrieves feedback records as a Pandas DataFrame with optional filters.
    """
    init_db(db_path)
    query = "SELECT * FROM feedback WHERE 1=1"
    params = []

    if sentiment_filter and sentiment_filter != "All":
        query += " AND sentiment = ?"
        params.append(sentiment_filter)

    if category_filter and category_filter != "All":
        query += " AND category = ?"
        params.append(category_filter)

    if source_filter and source_filter != "All":
        query += " AND source = ?"
        params.append(source_filter.lower())

    if search_keyword and search_keyword.strip():
        query += " AND (review_text LIKE ? OR customer_id LIKE ?)"
        like_pattern = f"%{search_keyword.strip()}%"
        params.extend([like_pattern, like_pattern])

    if start_date:
        query += " AND date(created_at) >= date(?)"
        params.append(str(start_date))

    if end_date:
        query += " AND date(created_at) <= date(?)"
        params.append(str(end_date))

    query += " ORDER BY created_at DESC"

    if limit and limit > 0:
        query += f" LIMIT {int(limit)}"

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def get_feedback_statistics(db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Computes key performance metrics and KPIs across all stored reviews.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "csat_score": 0.0,
                "avg_confidence": 0.0,
                "nps_indicator": 0.0
            }

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE sentiment = 'Positive'")
        positive = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE sentiment = 'Negative'")
        negative = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE sentiment = 'Neutral'")
        neutral = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM feedback")
        avg_conf = cursor.fetchone()[0] or 0.0

        pos_pct = round((positive / total) * 100, 1)
        neg_pct = round((negative / total) * 100, 1)
        neu_pct = round((neutral / total) * 100, 1)

        # Net Promoter Score proxy: % Positive - % Negative
        nps = round(pos_pct - neg_pct, 1)

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "csat_score": pos_pct,  # Customer Satisfaction %
            "avg_confidence": round(avg_conf * 100, 1),
            "nps_indicator": nps
        }


def delete_feedback(feedback_id: int, db_path: str = DB_PATH) -> bool:
    """Deletes a single feedback row by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_feedback(db_path: str = DB_PATH):
    """Clears all records from the feedback table."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback")
        conn.commit()


def seed_sample_data(force: bool = False, db_path: str = DB_PATH):
    """
    Seeds initial realistic customer feedback if database is empty or forced.
    Spans multiple dates and categories to provide rich immediate visualization.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback")
        count = cursor.fetchone()[0]

    if count > 0 and not force:
        return

    sample_reviews = [
        # Positive Reviews
        ("The quality of this product completely exceeded my expectations! Highly recommend to everyone.", "Positive", 0.94, 0.94, 0.02, 0.04, "Product Quality"),
        ("Customer support was incredibly helpful and solved my billing query within 5 minutes. Outstanding service!", "Positive", 0.97, 0.97, 0.01, 0.02, "Customer Service"),
        ("Fast delivery, well packaged, and works exactly as advertised. Will definitely buy again.", "Positive", 0.91, 0.91, 0.03, 0.06, "Delivery & Shipping"),
        ("Fantastic user interface! The dashboard is sleek, intuitive, and saves our team hours every week.", "Positive", 0.96, 0.96, 0.01, 0.03, "User Experience"),
        ("Great value for money. Compared to other competitors, this offers the best features at this price point.", "Positive", 0.89, 0.89, 0.04, 0.07, "Pricing & Billing"),
        ("Super happy with my purchase. The battery life is impressive and build quality is premium.", "Positive", 0.93, 0.93, 0.02, 0.05, "Product Quality"),
        ("The representative was courteous, polite, and resolved my issue on the very first call.", "Positive", 0.95, 0.95, 0.02, 0.03, "Customer Service"),
        ("Arrived two days earlier than expected. Seamless ordering process from start to finish.", "Positive", 0.92, 0.92, 0.03, 0.05, "Delivery & Shipping"),
        ("An absolute game changer for our daily workflow. Cannot imagine working without it now.", "Positive", 0.98, 0.98, 0.01, 0.01, "Product Quality"),
        ("Flawless experience. Setup took less than two minutes and everything synced automatically.", "Positive", 0.90, 0.90, 0.04, 0.06, "User Experience"),
        ("Brilliant update! The new reporting filters are crisp and responsive.", "Positive", 0.92, 0.92, 0.03, 0.05, "User Experience"),
        ("Love the sleek design and reliable performance. Worth every single penny.", "Positive", 0.94, 0.94, 0.02, 0.04, "Product Quality"),

        # Negative Reviews
        ("Terrible experience. The item stopped working after just three days of use. Extremely disappointed.", "Negative", 0.96, 0.01, 0.96, 0.03, "Product Quality"),
        ("Waited on hold for over 45 minutes and the agent was rude and unhelpful. Disgraceful customer support.", "Negative", 0.98, 0.01, 0.98, 0.01, "Customer Service"),
        ("Delivery was delayed by two weeks without any tracking update. Package arrived damaged.", "Negative", 0.94, 0.02, 0.94, 0.04, "Delivery & Shipping"),
        ("They charged my credit card twice for one subscription and refused to issue an immediate refund.", "Negative", 0.95, 0.01, 0.95, 0.04, "Pricing & Billing"),
        ("The software crashes constantly whenever I try to export data to CSV. Very frustrating bug.", "Negative", 0.92, 0.02, 0.92, 0.06, "User Experience"),
        ("Poor build material. Flimsy plastic and feels cheap. Definitely not worth the high price tag.", "Negative", 0.91, 0.03, 0.91, 0.06, "Product Quality"),
        ("Never received my order and support keeps giving generic automated replies. Worst experience ever.", "Negative", 0.97, 0.01, 0.97, 0.02, "Customer Service"),
        ("Hidden fees added at checkout that were never mentioned upfront. Completely dishonest pricing.", "Negative", 0.93, 0.02, 0.93, 0.05, "Pricing & Billing"),

        # Neutral Reviews
        ("The product is okay. It does the job, but nothing particularly impressive or groundbreaking.", "Neutral", 0.84, 0.08, 0.08, 0.84, "Product Quality"),
        ("Package arrived on the scheduled date. Standard cardboard box with standard contents.", "Neutral", 0.88, 0.05, 0.07, 0.88, "Delivery & Shipping"),
        ("Average service. The issue was solved eventually, though it took standard turnaround time.", "Neutral", 0.82, 0.10, 0.08, 0.82, "Customer Service"),
        ("Pricing is in line with the market average. Not cheap, but not overpriced either.", "Neutral", 0.86, 0.07, 0.07, 0.86, "Pricing & Billing"),
        ("Interface is functional. Takes some getting used to the navigation menu layout.", "Neutral", 0.80, 0.10, 0.10, 0.80, "User Experience"),
        ("Received confirmation email and tracking link as expected.", "Neutral", 0.89, 0.05, 0.06, 0.89, "Delivery & Shipping"),
        ("Features match the specifications listed in the user manual.", "Neutral", 0.85, 0.08, 0.07, 0.85, "Product Quality"),
        ("Service resumed after regular maintenance window concluded.", "Neutral", 0.87, 0.06, 0.07, 0.87, "User Experience"),
    ]

    base_time = datetime.now()
    records_to_insert = []
    for i, (text, sentiment, conf, p_pos, p_neg, p_neu, cat) in enumerate(sample_reviews):
        # Stagger dates over the last 14 days
        days_ago = (len(sample_reviews) - i) % 14
        review_time = (base_time - timedelta(days=days_ago, hours=i * 2, minutes=i * 11)).strftime("%Y-%m-%d %H:%M:%S")
        records_to_insert.append({
            "review_text": text,
            "cleaned_text": text.lower(),
            "sentiment": sentiment,
            "confidence": conf,
            "prob_positive": p_pos,
            "prob_negative": p_neg,
            "prob_neutral": p_neu,
            "category": cat,
            "source": "sample",
            "customer_id": f"CUST-{1000 + i}",
            "created_at": review_time
        })

    insert_bulk_feedback(records_to_insert, db_path=db_path)

