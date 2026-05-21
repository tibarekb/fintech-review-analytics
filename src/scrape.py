from google_play_scraper import app, reviews, Sort 
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

import string

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # required in NLTK 3.8+
nltk.download('omw-1.4', quiet=True)     # WordNet multilingual support


def scrape_fintech_reviews(app_dict, count=500):
    """ Fetches recent reviews for a dictionary of apps"""
    all_reviews = []
    for name, pkg in app_dict.items():
        try:
            res, _ = reviews(pkg, lang='en', country='et', sort= Sort.NEWEST, count=count)
            for r in res:
                all_reviews.append({
                    'app': name,
                    'review': r['content'],
                    'rating': r['score'],
                    'date': r['at'],
                    'thumbs': r['thumbsUpCount']
                })
            print(f"  Scraped {len(res)} reviews for {name}")
        except Exception as e:
            print(f"  Failed to scrape  {name}: {e}")
    return pd.DataFrame(all_reviews)

def missing_data(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    for col in df.columns:
        status = f"{missing[col]} missing ({missing_pct[col]}%)" if missing[col] > 0 else "OK"
        print(f"{col:<15}: {status}")

def duplicate_data(df):
    #Exact duplicates on review text
    exact_dupes = df.duplicated(subset=['review'], keep=False)
    print(f"Exact duplicate reviews: {exact_dupes.sum()}")

    #Empty reviews
    empty_reviews = (df['review'].str.strip() == '').sum()
    print(f"Empty reviews: {empty_reviews}")

def date_normalization(df):
    #Convert to pandas datetime then format as YYYY-MM-DD string
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    print("\nAfter Normalization:")
    print(df['date'].head(3).to_string())
    print(f"dtype: {df['date'].dtype}")

    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

def clean_text(text):
    """Standardize review text: collapse whitespace, strip edges"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/newlines
    text = text.strip()
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenizes text, removes stopwords, and applies lemmatization.
    Matches the notebook's modular_nlp_pipeline logic.
    """
    # 1. Cleaning (Regex)
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())

    # 2. Tokenization
    tokens = nltk.word_tokenize(text)

    # 3. Stop-word removal & Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(processed)







