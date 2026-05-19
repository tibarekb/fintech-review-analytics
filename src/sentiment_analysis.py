from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import logging
import nltk


# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def load_sentiment_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
     ERROR HANDLING: Safe model loading with fallback.
    """
    try:
        classifier = pipeline("sentiment-analysis", model=model_name)
        logging.info(f"Successfully loaded {model_name}")
        return classifier
    except Exception as e:
        logging.error(f"Failed to load transformer model {model_name}: {e}")
        return None

def analyze_sentiment_vader(text):
    """
    Lexicon-based sentiment with basic error handling.
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(str(text))['compound']
    except Exception:
        return 0.0

def analyze_sentiment_transformer(classifier, texts, batch_size=32):
    """
     ERROR HANDLING: Handles truncation and batch failures.
    """
    if classifier is None:
        return [{'label': 'UNAVAILABLE', 'score': 0.0}] * len(texts)
    
    # Transformer models have a 512 token limit. 
    # We truncate the raw string roughly to avoid index errors.
    truncated_texts = [str(t)[:512] for t in texts]
    
    try:
        return classifier(truncated_texts, batch_size=batch_size)
    except Exception as e:
        logging.warning(f"Batch analysis failed: {e}")
        return [{'label': 'ERROR', 'score': 0.0}] * len(texts)
