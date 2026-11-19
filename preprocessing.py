"""
Text Preprocessing Module for Sentiment Analysis System.

This module provides comprehensive NLP preprocessing utilities including:
- Contraction expansion (e.g. don't -> do not)
- Sentiment-aware stopword removal (preserves negation words like 'not', 'no', 'never')
- URL, HTML tag, and special character removal
- Tokenization and lemmatization/stemming
"""

import re
import string
import unicodedata
import warnings

# Suppress benign NLTK corpus directory permission warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nltk")

# Common English contractions dictionary
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

# Negation words to preserve during stopword filtering
NEGATION_WORDS = {
    "no", "not", "never", "none", "neither", "nor", "nowhere", "hardly",
    "scarcely", "barely", "cannot", "without", "against"
}

# Standard English stopwords fallback
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

# Filter out negations from stopwords so sentiment isn't corrupted
SENTIMENT_STOPWORDS = DEFAULT_STOPWORDS - NEGATION_WORDS

# Lazy-loaded NLTK resources
_nltk_initialized = False
_stemmer = None


def init_nltk():
    """Safely initialize NLTK stemmer and stopwords."""
    global _nltk_initialized, _stemmer, SENTIMENT_STOPWORDS
    if _nltk_initialized:
        return
    try:
        import nltk
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
        try:
            from nltk.corpus import stopwords
            nltk_stops = set(stopwords.words("english"))
            SENTIMENT_STOPWORDS = nltk_stops - NEGATION_WORDS
        except Exception:
            pass
    except ImportError:
        pass
    _nltk_initialized = True


def expand_contractions(text: str) -> str:
    """Expands contractions like don't to do not."""
    if not isinstance(text, str):
        return ""
    
    # Pattern to match contraction keys
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r')\b',
        flags=re.IGNORECASE
    )
    
    def replace(match):
        word = match.group(0).lower()
        expanded = CONTRACTION_MAP.get(word, word)
        if match.group(0)[0].isupper():
            expanded = expanded.capitalize()
        return expanded

    return pattern.sub(replace, text)


def clean_text(text: str, remove_stopwords: bool = True, stem: bool = False) -> str:
    """
    Cleans raw customer feedback text for machine learning.
    
    Steps:
    1. Handle non-string / NaN
    2. Normalize Unicode accents (e.g. é -> e)
    3. Expand contractions (e.g. wasn't -> was not)
    4. Remove URLs (http://, https://, www.*)
    5. Remove HTML tags (<p>, <br>, etc.)
    6. Remove email addresses
    7. Remove special characters and digits (keeping only alphabets and essential punctuation spaces)
    8. Lowercase
    9. Tokenize by whitespace
    10. Optionally remove non-negation stopwords
    11. Optionally apply Porter Stemming
    
    Args:
        text: Input review string
        remove_stopwords: Whether to strip neutral stopwords
        stem: Whether to apply Porter stemmer
        
    Returns:
        Cleaned, preprocessed text string
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    if not text:
        return ""
        
    init_nltk()

    # 1. Unicode normalization (NFKD)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # 2. Expand contractions
    text = expand_contractions(text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 5. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 6. Replace emojis / special symbols with space, keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 7. Convert to lowercase
    text = text.lower()

    # 8. Tokenize on whitespace
    tokens = text.split()

    # 9. Stopword removal (sentiment-aware)
    if remove_stopwords:
        tokens = [w for w in tokens if w not in SENTIMENT_STOPWORDS and len(w) > 1]

    # 10. Stemming (optional, PorterStemmer)
    if stem and _stemmer:
        tokens = [_stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


def get_text_statistics(text: str) -> dict:
    """
    Calculates basic descriptive statistics of a text.
    """
    if not isinstance(text, str):
        return {"char_count": 0, "word_count": 0, "avg_word_length": 0.0}
    
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_len = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": round(avg_len, 2)
    }
