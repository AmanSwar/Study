# Chapter 13: Alternative Data and NLP Signals

## Introduction

In Chapters 1-12, we built trading systems around traditional market data: OHLCV prices, technical indicators, and basic fundamental metrics. But **traditional data is already embedded into prices**. By the time financial statements are public or mainstream news is published, the market has already largely adjusted.

This is where **alternative data** enters the quantitative trader's toolkit. Alternative data consists of non-traditional information sources that may contain predictive signals before they're fully reflected in market prices:

- **News and social media sentiment**: What are people saying about a company? Is the tone positive or negative?
- **Earnings call transcripts**: What's the management tone? How transparent are they being?
- **SEC filings**: What changed since last quarter? Are they obfuscating information?
- **Web and social signals**: Google searches, job postings, satellite imagery, employee reviews
- **Behavioral signals**: Unusual trading patterns, insider buying, unusual options flow

The machine learning and NLP expertise you already have from earlier chapters is **perfectly suited** to extract signals from these unstructured data sources. We'll build:

1. **News sentiment pipelines** using FinBERT (BERT fine-tuned for financial text)
2. **Earnings analysis systems** that extract management tone and filing changes
3. **Alternative data integrations** for web scraping, social media, and other sources

All code will be production-ready with proper error handling, type hints, and docstrings. We'll show how to integrate these signals into your NSE trading system using Zerodha.

---

## Module 13.1: News Sentiment Analysis

### 13.1.1 The Case for News Sentiment

**Why does sentiment matter for trading?**

Markets are driven by expectations about future cash flows. News can shift those expectations instantly. Consider:

- A CEO announces unexpected insider buying → stock rallies
- A competitor launches a disruptive product → stock falls
- An earnings miss with bullish guidance → stock might rise (guidance expectations matter)

The **efficient market hypothesis** (EMH) suggests prices instantly incorporate all available information. But empirical research consistently shows:

- **Event drift**: Stock prices continue moving for days/weeks after major news (Bender et al., 2013)
- **Sentiment lag**: Markets take time to fully process news sentiment (Tetlock, 2007)
- **Anomalies in dissemination**: Local news drives local trading (Higgs & Worthington, 2008)

For Indian markets (NSE), this is **especially pronounced** because:
- Retail participation is growing (lower market efficiency)
- Information dissemination is slower for smaller stocks
- Hindi/regional language news often precedes English announcements

**The challenge**: Humans can't read 1000+ financial articles daily. Machine learning can.

### 13.1.2 Text Preprocessing for Financial News

Raw text requires cleaning before we can extract sentiment. This isn't just removing punctuation—financial text has special challenges.

#### Tokenization and Basic Cleaning

```python
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
import pandas as pd

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class FinancialTextCleaner:
    """
    Production-grade text preprocessor for financial news.
    
    Handles:
    - URL removal
    - HTML entity decoding
    - Currency/number normalization
    - Stop word removal
    - Lemmatization
    - Case normalization
    
    Attributes:
        stock_ticker: Company ticker (e.g., 'INFY') to preserve in text
        lemmatizer: NLTK lemmatizer instance
        stop_words: Set of English stop words
    """
    
    def __init__(self, stock_ticker: str = None):
        """
        Initialize the text cleaner.
        
        Args:
            stock_ticker: Stock ticker to preserve (not remove as stopword)
        """
        self.stock_ticker = stock_ticker
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Remove company ticker from stop words if provided
        if stock_ticker:
            self.stop_words.discard(stock_ticker.lower())
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    def remove_html_entities(self, text: str) -> str:
        """Decode HTML entities like &amp; → &"""
        import html
        return html.unescape(text)
    
    def normalize_numbers(self, text: str) -> str:
        """
        Normalize numbers while preserving context.
        
        Examples:
            "Rs. 1,000 crore" → "<NUMBER> crore"
            "1000% increase" → "<NUMBER>%"
            "Q3 2024 results" → "Q3 2024 results" (preserve quarter)
        """
        # Preserve quarters
        text = re.sub(r'\b[Qq]\d\s*[2-4]?\b', '<QUARTER>', text)
        
        # Normalize currency amounts (preserve currency context)
        text = re.sub(r'(?:Rs|₹|USD|\$|EUR)\s*\.?\s*[\d,\.]+\s*(?:crore|lakh|million|billion|thousand)?',
                      '<CURRENCY>', text)
        
        # Normalize remaining numbers
        text = re.sub(r'\b\d+(?:[,\.]\d+)*\b', '<NUMBER>', text)
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Uses NLTK punkt tokenizer which is trained on financial text.
        """
        return sent_tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str], keep_negations: bool = True) -> List[str]:
        """
        Remove stopwords with special handling for negations.
        
        In sentiment analysis, negations are CRITICAL:
        "not good" is very different from "good"
        
        Args:
            tokens: List of word tokens
            keep_negations: If True, preserve negation words (not, no, etc.)
        
        Returns:
            Filtered list of tokens
        """
        negation_words = {'not', 'no', 'nor', 'neither', 'never', 'nobody', 'nothing'}
        
        filtered = []
        for token in tokens:
            # Keep negations regardless of stopword status
            if keep_negations and token.lower() in negation_words:
                filtered.append(token)
            # Remove other stopwords
            elif token.lower() not in self.stop_words:
                filtered.append(token)
        
        return filtered
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens (reduce to base form).
        
        Examples:
            "running" → "run"
            "better" → "good"
            "earnings" → "earning"
        """
        return [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
    
    def clean_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Complete cleaning pipeline.
        
        Args:
            text: Raw text to clean
            remove_stopwords: Whether to remove stopwords
        
        Returns:
            Cleaned text
        """
        # Remove URLs and HTML
        text = self.remove_urls(text)
        text = self.remove_html_entities(text)
        
        # Normalize numbers
        text = self.normalize_numbers(text)
        
        # Basic cleaning
        text = text.strip()
        
        # Tokenize
        tokens = self.tokenize_words(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Rejoin
        return ' '.join(tokens)


# Example usage
cleaner = FinancialTextCleaner(stock_ticker='INFY')
raw_text = """
INFOSY reached Rs. 1,500 crore in Q3 2024!
See more: http://example.com
The company showed 25% growth in revenue.
"""
cleaned = cleaner.clean_text(raw_text)
print(f"Original: {raw_text}")
print(f"Cleaned: {cleaned}")
```

### 13.1.3 Dictionary-Based Sentiment: Loughran-McDonald Lexicon

Before diving into neural networks, let's understand the simpler approach: **dictionary-based sentiment**. The Loughran-McDonald financial sentiment dictionary is specifically built for financial text (unlike general sentiment dictionaries like VADER).

```python
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter

class LoughranMcdonaldSentiment:
    """
    Dictionary-based sentiment using Loughran-McDonald lexicon.
    
    The L-M lexicon contains ~4000 negative and ~355 positive words
    specific to financial contexts. Examples:
    
    Negative: "risks", "losses", "uncertainty", "decline"
    Positive: "gain", "increased", "profit", "strong"
    
    Advantages:
    - Fast (no neural network inference)
    - Interpretable (know exactly why text scored as positive/negative)
    - Domain-specific (built from 10-K filings)
    
    Disadvantages:
    - Misses context (e.g., "not good" scored as neutral)
    - Misses emerging sentiment terms
    - No intensity measure (all words weighted equally)
    """
    
    # Loughran-McDonald dictionaries (subset for example)
    # In production, load full list from official source
    POSITIVE_WORDS = {
        'strong', 'gain', 'increased', 'profit', 'exceed', 'growth',
        'improve', 'better', 'opportunity', 'positive', 'successful',
        'confident', 'excel', 'advance', 'highest', 'boost',
        'leading', 'success', 'outperform', 'outstanding'
    }
    
    NEGATIVE_WORDS = {
        'loss', 'weak', 'decline', 'risk', 'uncertain', 'negative',
        'challenge', 'below', 'worst', 'difficult', 'decrease',
        'falling', 'issue', 'problem', 'concern', 'worst', 'struggle',
        'obstacle', 'adversely', 'unsustainable', 'volatile'
    }
    
    UNCERTAINTY_WORDS = {
        'may', 'might', 'could', 'possibly', 'uncertain', 'risk',
        'fluctuate', 'volatility', 'pending', 'contingent'
    }
    
    LITIGIOUS_WORDS = {
        'litigation', 'legal', 'lawsuit', 'claim', 'court', 'alleged'
    }
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.dictionaries = {
            'positive': self.POSITIVE_WORDS,
            'negative': self.NEGATIVE_WORDS,
            'uncertainty': self.UNCERTAINTY_WORDS,
            'litigious': self.LITIGIOUS_WORDS
        }
    
    def calculate_sentiment(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate sentiment scores for token list.
        
        Args:
            tokens: List of word tokens (should be cleaned/lemmatized)
        
        Returns:
            Dictionary with scores:
            {
                'positive_count': int,
                'negative_count': int,
                'uncertainty_count': int,
                'litigious_count': int,
                'net_sentiment': float in [-1, 1],
                'sentiment_score': float in [-1, 1],
                'word_count': int
            }
        """
        token_lower = [t.lower() for t in tokens]
        
        pos_count = sum(1 for t in token_lower if t in self.POSITIVE_WORDS)
        neg_count = sum(1 for t in token_lower if t in self.NEGATIVE_WORDS)
        unc_count = sum(1 for t in token_lower if t in self.UNCERTAINTY_WORDS)
        lit_count = sum(1 for t in token_lower if t in self.LITIGIOUS_WORDS)
        
        total_sentiment_words = pos_count + neg_count
        word_count = len(tokens)
        
        # Net sentiment: positive minus negative
        if total_sentiment_words == 0:
            net_sentiment = 0.0
            sentiment_score = 0.0
        else:
            net_sentiment = (pos_count - neg_count) / total_sentiment_words
            sentiment_score = net_sentiment / (1 + unc_count)  # Penalize uncertainty
        
        return {
            'positive_count': pos_count,
            'negative_count': neg_count,
            'uncertainty_count': unc_count,
            'litigious_count': lit_count,
            'net_sentiment': net_sentiment,
            'sentiment_score': sentiment_score,
            'word_count': word_count,
            'sentiment_word_ratio': total_sentiment_words / max(word_count, 1)
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        End-to-end sentiment analysis for raw text.
        
        Args:
            text: Raw text to analyze
        
        Returns:
            Sentiment metrics dictionary
        """
        cleaner = FinancialTextCleaner()
        tokens = cleaner.tokenize_words(text)
        tokens = cleaner.remove_stopwords(tokens)
        
        return self.calculate_sentiment(tokens)


# Example
lm_sentiment = LoughranMcdonaldSentiment()

text1 = "Strong earnings growth exceeded expectations with positive outlook"
tokens1 = text1.split()
result1 = lm_sentiment.calculate_sentiment(tokens1)
print(f"Text: {text1}")
print(f"Sentiment: {result1['sentiment_score']:.3f}")
print(f"Positive: {result1['positive_count']}, Negative: {result1['negative_count']}\n")

text2 = "Revenue fell by 30% amid market uncertainty and litigation risks"
tokens2 = text2.split()
result2 = lm_sentiment.calculate_sentiment(tokens2)
print(f"Text: {text2}")
print(f"Sentiment: {result2['sentiment_score']:.3f}")
print(f"Positive: {result2['positive_count']}, Negative: {result2['negative_count']}")
```

### 13.1.4 Transformer-Based Sentiment: FinBERT

Dictionary methods are fast but limited. **FinBERT** is BERT (Bidirectional Encoder Representations from Transformers) fine-tuned on financial text. It understands context, negation, and domain-specific language.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Tuple
import numpy as np

class FinBERTSentiment:
    """
    Production-grade FinBERT sentiment analyzer.
    
    FinBERT is BERT fine-tuned on 10-K filings (financial language).
    It outputs scores for three classes: POSITIVE, NEGATIVE, NEUTRAL.
    
    Key advantages over dictionary methods:
    - Understands context ("not good" → negative, not neutral)
    - Captures nuance and domain-specific language
    - Pre-trained on financial documents
    - Handles out-of-vocabulary words via subword tokenization
    
    Mathematical foundation:
    For each input text, FinBERT outputs logits z = [z_pos, z_neg, z_neu]
    These are converted to probabilities via softmax:
    
    p_c = exp(z_c) / Σ_i exp(z_i)
    
    where c ∈ {POSITIVE, NEGATIVE, NEUTRAL}
    
    We convert to single sentiment score:
    sentiment = p_positive - p_negative
    
    This gives a score in [-1, 1] where:
    -1 = completely negative
    0 = neutral
    +1 = completely positive
    """
    
    def __init__(self, model_name: str = "yiyanghkust/finbert-pretrain", 
                 device: str = None, batch_size: int = 32):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' for GPU, 'cpu' for CPU. Auto-detects if None.
            batch_size: Batch size for inference
        
        Note:
            First run downloads ~500MB model. Subsequent runs use cache.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        print(f"Loading FinBERT from {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # positive, negative, neutral
        ).to(self.device)
        self.model.eval()  # Evaluation mode (no dropout, no batch norm updates)
        
        # Class mapping
        self.labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    
    def analyze_single(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Analyze sentiment of single text.
        
        Args:
            text: Input text (truncated to 512 tokens max)
            max_length: Max tokens (FinBERT trained on 512)
        
        Returns:
            {
                'sentiment': float in [-1, 1],
                'positive_score': float in [0, 1],
                'negative_score': float in [0, 1],
                'neutral_score': float in [0, 1],
                'predicted_label': str,
                'confidence': float in [0, 1]
            }
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # Shape: (3,)
        
        # Softmax to probabilities
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        
        # Sentiment score: positive - negative
        sentiment = float(probs[2] - probs[0])  # [2] = positive, [0] = negative
        
        # Predicted label
        pred_idx = int(torch.argmax(logits))
        pred_label = self.labels[pred_idx]
        
        return {
            'sentiment': sentiment,
            'positive_score': float(probs[2]),
            'neutral_score': float(probs[1]),
            'negative_score': float(probs[0]),
            'predicted_label': pred_label,
            'confidence': float(max(probs))
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # Shape: (batch_size, 3)
            
            # Process each item in batch
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # Shape: (batch_size, 3)
            
            for j, text in enumerate(batch):
                sentiment = float(probs[j, 2] - probs[j, 0])  # positive - negative
                pred_idx = int(np.argmax(probs[j]))
                
                results.append({
                    'sentiment': sentiment,
                    'positive_score': float(probs[j, 2]),
                    'neutral_score': float(probs[j, 1]),
                    'negative_score': float(probs[j, 0]),
                    'predicted_label': self.labels[pred_idx],
                    'confidence': float(max(probs[j]))
                })
        
        return results


# Example usage
finbert = FinBERTSentiment(device='cpu')  # Use 'cuda' if GPU available

texts = [
    "Strong earnings with positive guidance",
    "Revenue fell amid market uncertainty",
    "Stable performance consistent with expectations",
    "Record profits and market share gains"
]

results = finbert.analyze_batch(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"  Sentiment: {result['sentiment']:+.3f} | Label: {result['predicted_label']}")
    print(f"  Scores - Pos: {result['positive_score']:.3f}, "
          f"Neu: {result['neutral_score']:.3f}, "
          f"Neg: {result['negative_score']:.3f}\n")
```

### 13.1.5 Aggregating Article Sentiment to Trading Signals

Now we have sentiment scores for individual articles. How do we convert this into actionable trading signals?

```python
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

class NewsSignalGenerator:
    """
    Convert article-level sentiment to stock-level daily signals.
    
    Process:
    1. Collect all articles for a stock on a given day
    2. Aggregate sentiment scores (mean, median, weighted, etc.)
    3. Compare to rolling baseline
    4. Generate trading signal
    
    Mathematical framework:
    
    For stock s on date t with articles A:
    
    Daily sentiment:
    S_t = (1/|A|) * Σ_i sentiment_i  (mean aggregation)
    
    Or weighted by recency (more recent = higher weight):
    S_t = Σ_i (w_i * sentiment_i) / Σ_i w_i
    where w_i = exp(-λ * (t - t_i)) / (24 hours)
    
    Signal generation:
    z_t = (S_t - μ_baseline) / σ_baseline
    
    where μ_baseline, σ_baseline are computed from historical sentiment
    
    Trading signal (z-score):
    signal = z_t if |z_t| > threshold else 0
    """
    
    def __init__(self, baseline_days: int = 30, z_score_threshold: float = 1.0):
        """
        Initialize signal generator.
        
        Args:
            baseline_days: Days to compute rolling baseline sentiment
            z_score_threshold: Z-score threshold for signal generation
        """
        self.baseline_days = baseline_days
        self.z_score_threshold = z_score_threshold
    
    def aggregate_daily_sentiment(
        self,
        articles_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate article sentiments to daily level.
        
        Args:
            articles_df: DataFrame with columns:
                - 'timestamp': datetime of publication
                - 'sentiment': float sentiment score
                - 'confidence': float confidence in prediction
        
        Returns:
            {
                'mean_sentiment': float,
                'median_sentiment': float,
                'weighted_sentiment': float (weighted by recency),
                'max_sentiment': float,
                'min_sentiment': float,
                'article_count': int,
                'high_confidence_sentiment': float (sentiment of high confidence articles)
            }
        """
        if len(articles_df) == 0:
            return {
                'mean_sentiment': 0.0,
                'median_sentiment': 0.0,
                'weighted_sentiment': 0.0,
                'max_sentiment': 0.0,
                'min_sentiment': 0.0,
                'article_count': 0,
                'high_confidence_sentiment': 0.0
            }
        
        sentiments = articles_df['sentiment'].values
        confidences = articles_df['confidence'].values
        timestamps = articles_df['timestamp'].values
        
        # Time decay weights (more recent articles weighted higher)
        now = pd.Timestamp.now()
        time_diffs = (now - pd.to_datetime(timestamps)).total_seconds() / 3600  # hours
        recency_weights = np.exp(-0.1 * np.clip(time_diffs, 0, None))  # λ = 0.1
        
        # Normalize weights
        recency_weights = recency_weights / recency_weights.sum()
        
        # High confidence: confidence > 0.75
        high_conf_mask = confidences > 0.75
        
        return {
            'mean_sentiment': float(sentiments.mean()),
            'median_sentiment': float(np.median(sentiments)),
            'weighted_sentiment': float((sentiments * recency_weights).sum()),
            'max_sentiment': float(sentiments.max()),
            'min_sentiment': float(sentiments.min()),
            'article_count': len(articles_df),
            'high_confidence_sentiment': float(
                sentiments[high_conf_mask].mean() if high_conf_mask.any() else 0.0
            )
        }
    
    def compute_baseline_stats(
        self,
        historical_sentiments: pd.Series
    ) -> Tuple[float, float]:
        """
        Compute baseline mean and std from historical sentiments.
        
        Args:
            historical_sentiments: Series of daily sentiment scores
        
        Returns:
            (mean, std_dev)
        """
        return (float(historical_sentiments.mean()), 
                float(historical_sentiments.std()))
    
    def generate_signal(
        self,
        daily_sentiment: float,
        baseline_mean: float,
        baseline_std: float
    ) -> Dict[str, float]:
        """
        Generate trading signal from daily sentiment.
        
        Args:
            daily_sentiment: Aggregated sentiment for the day
            baseline_mean: Historical mean sentiment
            baseline_std: Historical std sentiment
        
        Returns:
            {
                'z_score': float,
                'signal': float (in [-1, 1] range),
                'strength': float (0 to 1, confidence in signal)
            }
        """
        if baseline_std < 1e-6:
            baseline_std = 1.0
        
        # Z-score normalization
        z_score = (daily_sentiment - baseline_mean) / baseline_std
        
        # Signal: only trade if Z-score exceeds threshold
        if abs(z_score) > self.z_score_threshold:
            signal = np.sign(z_score) * min(abs(z_score) / 3.0, 1.0)  # Clip to [-1, 1]
        else:
            signal = 0.0
        
        # Strength: how confident are we in this signal
        strength = min(abs(z_score) / 3.0, 1.0)
        
        return {
            'z_score': float(z_score),
            'signal': float(signal),
            'strength': float(strength)
        }
    
    def process_daily_news(
        self,
        articles_df: pd.DataFrame,
        historical_sentiment_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        End-to-end processing: articles → trading signal
        
        Args:
            articles_df: Today's articles with sentiment scores
            historical_sentiment_df: Historical daily sentiments
                DataFrame with columns: ['date', 'daily_sentiment']
        
        Returns:
            Complete signal dictionary
        """
        # Aggregate article-level to daily-level
        daily_agg = self.aggregate_daily_sentiment(articles_df)
        
        # Get baseline from historical data
        baseline_mean, baseline_std = self.compute_baseline_stats(
            historical_sentiment_df['daily_sentiment']
        )
        
        # Generate signal
        signal_dict = self.generate_signal(
            daily_agg['weighted_sentiment'],
            baseline_mean,
            baseline_std
        )
        
        # Combine
        return {**daily_agg, **signal_dict}


# Example
news_signal_gen = NewsSignalGenerator(baseline_days=30, z_score_threshold=1.0)

# Simulated articles for today
today_articles = pd.DataFrame({
    'timestamp': [
        datetime.now() - timedelta(hours=4),
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=1),
        datetime.now() - timedelta(hours=0.5),
    ],
    'sentiment': [0.65, 0.72, 0.58, 0.80],
    'confidence': [0.92, 0.88, 0.95, 0.90]
})

# Historical baseline (last 30 days)
np.random.seed(42)
historical_sentiments = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30),
    'daily_sentiment': np.random.normal(0.0, 0.2, 30)  # Mean ~0, Std ~0.2
})

result = news_signal_gen.process_daily_news(today_articles, historical_sentiments)
print(f"Daily Aggregated Sentiment: {result['mean_sentiment']:.3f}")
print(f"Weighted Sentiment (recency): {result['weighted_sentiment']:.3f}")
print(f"Article Count: {result['article_count']}")
print(f"Z-Score: {result['z_score']:.3f}")
print(f"Trading Signal: {result['signal']:+.3f}")
print(f"Signal Strength: {result['strength']:.3f}")
```

### 13.1.6 Event Studies: Measuring Price Reaction

Event study methodology measures how prices react to specific news events. This is both a validation technique (does your sentiment analysis actually predict prices?) and a signal generation technique.

```python
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple

class EventStudy:
    """
    Event study analysis: measure abnormal returns around news events.
    
    Mathematical framework:
    
    For stock s, event date t with event surprise E:
    
    Expected return (from market model):
    E[R_s,t] = α_s + β_s * R_market,t
    
    Actual return:
    R_s,t = realized return on date t
    
    Abnormal return:
    AR_s,t = R_s,t - E[R_s,t] = R_s,t - (α_s + β_s * R_market,t)
    
    Cumulative abnormal return (window [-n, +m] around event):
    CAR = Σ_i AR_i for i in [-n, +m]
    
    Statistical test (t-test):
    t = CAR / (σ_AR * sqrt(N))
    
    If |t| > t_critical, event has significant price impact
    """
    
    def __init__(self, estimation_window: int = 120):
        """
        Initialize event study.
        
        Args:
            estimation_window: Days for estimating market model (α, β)
        """
        self.estimation_window = estimation_window
    
    def estimate_market_model(
        self,
        stock_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate market model: R_stock = α + β * R_market
        
        Using OLS regression:
        (α, β) = argmin Σ(R_stock - α - β*R_market)^2
        
        Args:
            stock_returns: Array of daily stock returns
            market_returns: Array of daily market returns
        
        Returns:
            (alpha, beta) parameters
        """
        X = np.column_stack([np.ones(len(market_returns)), market_returns])
        params = np.linalg.lstsq(X, stock_returns, rcond=None)[0]
        return float(params[0]), float(params[1])
    
    def calculate_abnormal_returns(
        self,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        alpha: float,
        beta: float
    ) -> np.ndarray:
        """
        Calculate abnormal returns.
        
        AR_t = R_stock,t - (α + β * R_market,t)
        """
        expected_returns = alpha + beta * market_returns
        abnormal_returns = stock_returns - expected_returns
        return abnormal_returns
    
    def calculate_car(
        self,
        abnormal_returns: np.ndarray,
        window: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculate cumulative abnormal return (CAR).
        
        Args:
            abnormal_returns: Array of abnormal returns
            window: (start_day, end_day) relative to event (0 = event day)
                    Example: (-2, +2) = 2 days before to 2 days after
        
        Returns:
            (car_value, standard_error)
        """
        start, end = window
        # Map relative days to indices (if event is at index e)
        # We assume abnormal_returns is centered at event date
        
        car = abnormal_returns[start:end+1].sum()
        se = abnormal_returns.std()  # Simplified
        
        return float(car), float(se)
    
    def test_significance(
        self,
        car: float,
        se: float,
        n_obs: int,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Statistical test of CAR significance.
        
        H0: CAR = 0 (no abnormal returns)
        H1: CAR ≠ 0 (significant abnormal returns)
        
        t-statistic: t = CAR / (SE * sqrt(N))
        
        Args:
            car: Cumulative abnormal return
            se: Standard error of abnormal returns
            n_obs: Number of observations
            alpha: Significance level
        
        Returns:
            {
                't_statistic': float,
                'p_value': float,
                'significant': bool,
                'confidence_level': float (e.g., 0.95 for 95%)
            }
        """
        if se < 1e-6:
            se = 1e-6
        
        t_stat = car / (se * np.sqrt(n_obs))
        df = n_obs - 1
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        is_significant = p_value < alpha
        confidence = 1 - p_value
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(is_significant),
            'confidence_level': float(confidence)
        }
    
    def analyze_event(
        self,
        stock_returns_pre: np.ndarray,
        market_returns_pre: np.ndarray,
        stock_returns_event: np.ndarray,
        market_returns_event: np.ndarray,
        event_window: Tuple[int, int] = (-5, 5),
        sentiment_magnitude: float = None
    ) -> Dict[str, float]:
        """
        Complete event study analysis.
        
        Args:
            stock_returns_pre: Pre-event returns (for estimation)
            market_returns_pre: Pre-event market returns
            stock_returns_event: Returns around event
            market_returns_event: Market returns around event
            event_window: Window around event date
            sentiment_magnitude: Optional sentiment score (for correlation)
        
        Returns:
            Complete event study results
        """
        # Estimate market model on pre-event data
        alpha, beta = self.estimate_market_model(stock_returns_pre, market_returns_pre)
        
        # Calculate abnormal returns around event
        ar = self.calculate_abnormal_returns(
            stock_returns_event,
            market_returns_event,
            alpha,
            beta
        )
        
        # Calculate CAR
        car, se = self.calculate_car(ar, event_window)
        
        # Test significance
        test_results = self.test_significance(car, se, len(ar))
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'car': float(car),
            'abnormal_return_std': float(ar.std()),
            **test_results
        }


# Example: Event study for earnings surprise
np.random.seed(42)

# Pre-event data (120 days before earnings)
stock_returns_pre = np.random.normal(0.001, 0.02, 120)
market_returns_pre = np.random.normal(0.0005, 0.015, 120)

# Event window (±5 days around earnings, day 0 = earnings date)
stock_returns_event = np.random.normal(0.003, 0.02, 11)
market_returns_event = np.random.normal(0.0005, 0.015, 11)

# Simulate positive earnings surprise
stock_returns_event[5] = 0.05  # Large positive return on earnings day

event_study = EventStudy(estimation_window=120)
results = event_study.analyze_event(
    stock_returns_pre,
    market_returns_pre,
    stock_returns_event,
    market_returns_event,
    event_window=(-2, 2)
)

print("Event Study Results:")
print(f"  Market Model: R = {results['alpha']:.4f} + {results['beta']:.3f}*R_market")
print(f"  CAR ([-2, +2]): {results['car']:.4f}")
print(f"  t-statistic: {results['t_statistic']:.3f}")
print(f"  p-value: {results['p_value']:.4f}")
print(f"  Significant (p<0.05): {results['significant']}")
```

---

## Module 13.2: Earnings Call and Filing Analysis

### 13.2.1 Why Earnings Calls Matter

Earnings calls are when management discusses quarterly results with investors. Unlike earnings reports (10-Q) which are formulaic, calls contain:

- **Management tone**: Optimistic or pessimistic about future?
- **Key metrics discussion**: Which metrics get emphasized?
- **Guidance quality**: Is management confident in forward guidance?
- **Language patterns**: Complex, obfuscating language vs. clear communication

**Academic finding** (Larcker & Zakolyukina, 2012): Obfuscated language in earnings calls predicts lower future returns. Why? Managers obfuscate when they're hiding bad news.

### 13.2.2 Text Sources and Extraction

```python
import pandas as pd
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import re

class EarningsDataExtractor:
    """
    Extract earnings call transcripts and filings.
    
    Sources:
    1. Company websites (investor relations pages)
    2. Financial data providers (yfinance, rapidapi)
    3. SEC EDGAR (for US companies)
    4. BSE/NSE disclosures (for Indian companies)
    5. Stocktwits, financial news sites
    
    For NSE stocks, primary sources:
    - Company website IR section
    - BSE/NSE official announcements
    - Stock exchange filings
    """
    
    def __init__(self, data_provider: str = 'local'):
        """
        Initialize extractor.
        
        Args:
            data_provider: 'local' (files), 'web' (scrape), 'api' (paid service)
        """
        self.data_provider = data_provider
    
    def extract_from_file(self, filepath: str) -> str:
        """
        Read earnings transcript from file.
        
        Args:
            filepath: Path to text file containing transcript
        
        Returns:
            Full transcript text
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_earnings_transcript(self, transcript_text: str) -> Dict[str, str]:
        """
        Parse earnings transcript into sections.
        
        Typical structure:
        1. Opening remarks
        2. Management discussion
        3. Q&A session
        4. Closing remarks
        
        Args:
            transcript_text: Raw transcript text
        
        Returns:
            {
                'opening_remarks': str,
                'management_discussion': str,
                'qa_session': str,
                'closing': str
            }
        """
        sections = {
            'opening_remarks': '',
            'management_discussion': '',
            'qa_session': '',
            'closing': ''
        }
        
        # Split by common markers
        lines = transcript_text.split('\n')
        
        # Simple heuristic parsing (in production, use more sophisticated NLP)
        current_section = 'opening_remarks'
        buffer = []
        
        for line in lines:
            lower_line = line.lower()
            
            # Detect section changes
            if 'questions and answers' in lower_line or 'q&a' in lower_line:
                sections[current_section] = '\n'.join(buffer)
                current_section = 'qa_session'
                buffer = []
            elif 'closing' in lower_line or 'forward-looking' in lower_line:
                sections[current_section] = '\n'.join(buffer)
                current_section = 'closing'
                buffer = []
            else:
                buffer.append(line)
        
        # Capture remaining
        sections[current_section] = '\n'.join(buffer)
        
        return sections


class ManagementToneAnalyzer:
    """
    Analyze management tone from earnings transcripts.
    
    Key metrics:
    1. Positive vs negative word usage
    2. Confidence language (certain vs uncertain)
    3. Emphasis patterns (what gets repeated)
    4. Future orientation (discussing growth opportunities)
    """
    
    # Tone lexicons
    CONFIDENT_WORDS = {
        'confident', 'strong', 'significant', 'leading', 'well-positioned',
        'expect', 'believe', 'confident', 'momentum', 'growing', 'expanding'
    }
    
    UNCERTAIN_WORDS = {
        'uncertain', 'challenging', 'difficult', 'risk', 'may', 'could',
        'might', 'volatile', 'pending', 'unpredictable', 'fluctuate'
    }
    
    BULLISH_WORDS = {
        'growth', 'expansion', 'opportunity', 'upside', 'outperform',
        'improve', 'advance', 'innovation', 'competitive', 'leadership'
    }
    
    BEARISH_WORDS = {
        'decline', 'weakness', 'pressure', 'headwind', 'challenge',
        'competition', 'saturation', 'risks', 'downside', 'loss'
    }
    
    def __init__(self):
        """Initialize tone analyzer."""
        self.dictionaries = {
            'confident': self.CONFIDENT_WORDS,
            'uncertain': self.UNCERTAIN_WORDS,
            'bullish': self.BULLISH_WORDS,
            'bearish': self.BEARISH_WORDS
        }
    
    def analyze_tone(self, text: str) -> Dict[str, float]:
        """
        Analyze management tone from text.
        
        Args:
            text: Earnings transcript text
        
        Returns:
            {
                'confidence_score': float,
                'bullish_score': float,
                'word_counts': {dict of counts},
                'tone_summary': str
            }
        """
        tokens = text.lower().split()
        
        counts = {key: 0 for key in self.dictionaries}
        
        for token in tokens:
            for tone_type, words in self.dictionaries.items():
                if token in words:
                    counts[tone_type] += 1
        
        total_words = len(tokens)
        
        # Normalize
        confidence_score = (counts['confident'] - counts['uncertain']) / max(1, total_words)
        bullish_score = (counts['bullish'] - counts['bearish']) / max(1, total_words)
        
        return {
            'confidence_score': float(confidence_score),
            'bullish_score': float(bullish_score),
            'word_counts': counts,
            'total_words': total_words
        }


# Example usage
extractor = EarningsDataExtractor()
tone_analyzer = ManagementToneAnalyzer()

# Simulated earnings transcript
sample_transcript = """
Good morning everyone. Q3 was a strong quarter with confident outlook.
We're well-positioned for growth and expansion into new markets.
Despite some uncertain macroeconomic headwinds, our momentum remains bullish.
We expect significant opportunities in the coming quarters.
"""

tone_result = tone_analyzer.analyze_tone(sample_transcript)
print(f"Confidence Score: {tone_result['confidence_score']:.4f}")
print(f"Bullish Score: {tone_result['bullish_score']:.4f}")
print(f"Word Counts: {tone_result['word_counts']}")
```

### 13.2.3 Readability and Obfuscation Metrics

```python
import re
from typing import Dict

class FilingReadabilityAnalyzer:
    """
    Measure readability and obfuscation in SEC/BSE filings.
    
    Key insight (Larcker & Zakolyukina 2012):
    Managers obfuscate when hiding bad news. Metrics:
    - Flesch-Kincaid readability
    - FOG index (Gunning FOG)
    - Average word length
    - Sentence length
    
    Math:
    FOG = 0.4 * (words/sentences + 100 * complex_words/words)
    
    where complex_words = words with 3+ syllables
    
    Interpretation:
    - FOG < 12: Easy to read (transparent)
    - FOG > 15: Difficult to read (potentially obfuscated)
    """
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def count_syllables(self, word: str) -> int:
        """
        Estimate syllable count using heuristics.
        
        Rules:
        - Each vowel group = 1 syllable
        - Silent 'e' doesn't count
        - 'le' at end = 1 syllable
        """
        word = word.lower()
        
        # Remove non-alpha
        word = re.sub(r'[^a-z]', '', word)
        
        if len(word) <= 3:
            return 1
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjustments
        if word.endswith('e'):
            syllable_count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        
        return max(1, syllable_count)
    
    def calculate_fog_index(self, text: str) -> Dict[str, float]:
        """
        Calculate Gunning FOG readability index.
        
        FOG = 0.4 * (words/sentences + 100 * complex_words/words)
        
        Higher FOG = harder to read
        
        Args:
            text: Input text
        
        Returns:
            {
                'fog_index': float,
                'flesch_kincaid_grade': float,
                'avg_words_per_sentence': float,
                'avg_syllables_per_word': float,
                'complex_word_ratio': float
            }
        """
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        n_sentences = max(len(sentences), 1)
        
        # Word count and tokenization
        words = re.findall(r'\b[a-z]+\b', text.lower())
        n_words = len(words)
        
        if n_words == 0:
            return {
                'fog_index': 0.0,
                'flesch_kincaid_grade': 0.0,
                'avg_words_per_sentence': 0.0,
                'avg_syllables_per_word': 0.0,
                'complex_word_ratio': 0.0
            }
        
        # Syllable count
        total_syllables = sum(self.count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / n_words
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for word in words if self.count_syllables(word) >= 3)
        complex_word_ratio = complex_words / n_words
        
        # FOG index
        fog = 0.4 * (n_words / n_sentences + 100 * complex_word_ratio)
        
        # Flesch-Kincaid Grade
        fk_grade = (0.39 * n_words / n_sentences + 
                   11.8 * total_syllables / n_words - 15.59)
        
        return {
            'fog_index': float(fog),
            'flesch_kincaid_grade': float(fk_grade),
            'avg_words_per_sentence': float(n_words / n_sentences),
            'avg_syllables_per_word': float(avg_syllables_per_word),
            'complex_word_ratio': float(complex_word_ratio)
        }
    
    def analyze_filing(self, filing_text: str) -> Dict[str, float]:
        """
        Complete filing analysis.
        
        Args:
            filing_text: 10-K or 10-Q filing text
        
        Returns:
            Readability metrics
        """
        metrics = self.calculate_fog_index(filing_text)
        
        # Interpretation
        if metrics['fog_index'] > 18:
            clarity = 'Very difficult (potentially obfuscated)'
        elif metrics['fog_index'] > 14:
            clarity = 'Difficult (likely obfuscated)'
        elif metrics['fog_index'] > 12:
            clarity = 'Moderate'
        else:
            clarity = 'Easy (transparent)'
        
        metrics['clarity_assessment'] = clarity
        
        return metrics


# Example
analyzer = FilingReadabilityAnalyzer()

# Sample filing excerpt
sample_filing = """
The Company's operating results are subject to various risks and uncertainties.
Economic conditions and market volatility may have significant adverse effects on our business.
We face intense competitive pressures in our primary markets.
Future performance will depend on our ability to maintain technological leadership.
"""

results = analyzer.analyze_filing(sample_filing)
print(f"FOG Index: {results['fog_index']:.2f}")
print(f"Flesch-Kincaid Grade: {results['flesch_kincaid_grade']:.2f}")
print(f"Clarity: {results['clarity_assessment']}")
print(f"Avg Words/Sentence: {results['avg_words_per_sentence']:.2f}")
print(f"Complex Word Ratio: {results['complex_word_ratio']:.4f}")
```

### 13.2.4 Filing Change Detection

```python
from difflib import SequenceMatcher
from typing import List, Tuple

class FilingChangeDetector:
    """
    Detect changes between consecutive SEC/BSE filings.
    
    Key insight:
    What management chooses to emphasize or de-emphasize signals their views.
    
    Example:
    - Risk section grows significantly → management concerned
    - Revenue guidance section becomes vaguer → confidence dropping
    - New market opportunity section added → growth plans
    """
    
    def extract_section(self, filing_text: str, section_name: str) -> str:
        """
        Extract specific section from filing.
        
        Common sections in 10-K:
        - Item 1: Business
        - Item 1A: Risk Factors
        - Item 7: MD&A (Management Discussion & Analysis)
        - Item 8: Financial Statements
        
        Args:
            filing_text: Full filing text
            section_name: Section to extract (e.g., "Risk Factors", "MD&A")
        
        Returns:
            Section text
        """
        # Find section start
        pattern = rf"(?:Item\s+[0-9A-Z]+:|{re.escape(section_name)})"
        matches = list(re.finditer(pattern, filing_text, re.IGNORECASE))
        
        if not matches:
            return ""
        
        start = matches[0].start()
        
        # Find next section (or end)
        if len(matches) > 1:
            end = matches[1].start()
        else:
            end = len(filing_text)
        
        return filing_text[start:end]
    
    def compare_sections(self, old_text: str, new_text: str) -> Dict[str, float]:
        """
        Compare two versions of a filing section.
        
        Args:
            old_text: Previous filing section
            new_text: Current filing section
        
        Returns:
            {
                'similarity_ratio': float in [0, 1],
                'length_change_ratio': float,
                'added_content_ratio': float,
                'removed_content_ratio': float
            }
        """
        # Similarity ratio (0 = completely different, 1 = identical)
        matcher = SequenceMatcher(None, old_text, new_text)
        similarity = matcher.ratio()
        
        # Length changes
        old_length = len(old_text.split())
        new_length = len(new_text.split())
        length_change = (new_length - old_length) / max(old_length, 1)
        
        # Find added/removed content
        old_words = set(old_text.lower().split())
        new_words = set(new_text.lower().split())
        
        added_words = new_words - old_words
        removed_words = old_words - new_words
        
        added_ratio = len(added_words) / max(len(new_words), 1)
        removed_ratio = len(removed_words) / max(len(old_words), 1)
        
        return {
            'similarity_ratio': float(similarity),
            'length_change_ratio': float(length_change),
            'added_content_ratio': float(added_ratio),
            'removed_content_ratio': float(removed_ratio)
        }
    
    def detect_major_changes(
        self,
        old_filing: str,
        new_filing: str,
        section: str = "Risk Factors"
    ) -> Dict[str, float]:
        """
        Detect if major changes occurred.
        
        Signals of significant change:
        - Risk section grew >30%: new risks emerged
        - Risk section shrunk >20%: previous risks resolved
        - MD&A tone shifted: guidance changed
        
        Args:
            old_filing: Previous filing
            new_filing: Current filing
            section: Section to compare
        
        Returns:
            Change metrics
        """
        old_section = self.extract_section(old_filing, section)
        new_section = self.extract_section(new_filing, section)
        
        changes = self.compare_sections(old_section, new_section)
        
        # Interpret changes
        if changes['length_change_ratio'] > 0.30:
            interpretation = f"Risk section EXPANDED by {changes['length_change_ratio']*100:.1f}% - New concerns"
        elif changes['length_change_ratio'] < -0.20:
            interpretation = f"Risk section CONTRACTED by {abs(changes['length_change_ratio'])*100:.1f}% - Risks resolved"
        else:
            interpretation = "No major section changes"
        
        changes['interpretation'] = interpretation
        
        return changes


# Example
detector = FilingChangeDetector()

old_10k = """
Item 1A: Risk Factors

We face significant competitive pressures in our core markets.
Supply chain disruptions could impact our operations.
Regulatory changes may affect our business model.
"""

new_10k = """
Item 1A: Risk Factors

We face significant competitive pressures in our core markets.
Supply chain disruptions could impact our operations.
Regulatory changes may affect our business model.
Geopolitical tensions create new export barriers.
Cybersecurity breaches pose increasing risk.
Raw material costs are volatile and unpredictable.
"""

changes = detector.detect_major_changes(old_10k, new_10k, section="Risk Factors")
print(f"Section similarity: {changes['similarity_ratio']:.1%}")
print(f"Length change: {changes['length_change_ratio']:+.1%}")
print(f"Interpretation: {changes['interpretation']}")
```

---

## Module 13.3: Other Alternative Data Sources

### 13.3.1 Web Scraping for Financial Data

```python
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from datetime import datetime

class FinancialWebScraper:
    """
    Scrape financial data from websites.
    
    IMPORTANT LEGAL/ETHICAL CONSIDERATIONS:
    
    1. Check robots.txt and Terms of Service
    2. Respect rate limits (1-2 requests per second)
    3. Use User-Agent header (identify yourself)
    4. Don't scrape if API available
    5. For Indian sites: check with NSE/BSE before scraping
    6. Some sites explicitly forbid scraping in ToS
    
    Safe alternatives to scraping:
    - Official APIs (NSE, BSE provide APIs)
    - Zerodha API (integrated with your trading system)
    - Free data providers (yfinance, etc.)
    - Official data feeds (paid services)
    """
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            rate_limit_delay: Seconds between requests (be respectful!)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        # Identify yourself properly
        self.session.headers.update({
            'User-Agent': 'Financial-Analysis-Bot/1.0 (Contact: your@email.com)'
        })
    
    def check_robots_txt(self, domain: str) -> bool:
        """
        Check if scraping is allowed.
        
        Args:
            domain: Website domain (e.g., 'example.com')
        
        Returns:
            True if scraping likely allowed, False otherwise
        """
        try:
            url = f"https://{domain}/robots.txt"
            response = requests.get(url, timeout=5)
            
            # If it returns 200, read it
            if response.status_code == 200:
                # Check if our User-Agent is explicitly disallowed
                if 'User-agent: *' in response.text or 'User-agent: Financial' in response.text:
                    if 'Disallow: /' in response.text:
                        return False
            
            return True
        except:
            # If we can't check, assume caution
            return False
    
    def scrape_financial_page(self, url: str) -> BeautifulSoup:
        """
        Scrape a financial website.
        
        Args:
            url: URL to scrape
        
        Returns:
            BeautifulSoup object
        
        Raises:
            Exception if scraping fails
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            raise Exception(f"Failed to scrape {url}: {e}")
    
    def extract_financial_data(
        self,
        soup: BeautifulSoup,
        selectors: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Extract data using CSS selectors.
        
        Args:
            soup: BeautifulSoup object
            selectors: Dict of {field_name: css_selector}
                Example: {'price': '.stock-price', 'volume': '.volume'}
        
        Returns:
            Extracted data
        """
        data = {}
        
        for field, selector in selectors.items():
            try:
                element = soup.select_one(selector)
                if element:
                    data[field] = element.get_text().strip()
                else:
                    data[field] = None
            except:
                data[field] = None
        
        return data


class StockTwitsScraperExample:
    """
    Example: Extract sentiment from StockTwits (social trading platform).
    
    StockTwits provides API (https://api.stocktwits.com/api/2/streams/...)
    This example shows how to use their public API (better than scraping).
    """
    
    @staticmethod
    def get_sentiment_feed(ticker: str, limit: int = 30) -> List[Dict]:
        """
        Get recent sentiment posts for a ticker.
        
        Args:
            ticker: Stock ticker (e.g., 'INFY')
            limit: Number of posts to retrieve
        
        Returns:
            List of posts with sentiment
        """
        # Using public API (no scraping needed!)
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        
        try:
            response = requests.get(url, params={'limit': limit})
            response.raise_for_status()
            
            data = response.json()
            messages = data.get('messages', [])
            
            posts = []
            for msg in messages:
                posts.append({
                    'author': msg.get('user', {}).get('username'),
                    'message': msg.get('body'),
                    'sentiment': msg.get('sentiment'),  # 'Bullish' or 'Bearish'
                    'created_at': msg.get('created_at'),
                    'likes': msg.get('likes', {}).get('total', 0)
                })
            
            return posts
        
        except Exception as e:
            print(f"Error fetching StockTwits data: {e}")
            return []


# Example
stocktwits = StockTwitsScraperExample()
posts = stocktwits.get_sentiment_feed('INFY', limit=10)

print(f"Recent StockTwits posts for INFY:")
for post in posts[:3]:
    print(f"  {post['author']}: {post['message'][:50]}...")
    print(f"    Sentiment: {post['sentiment']}, Likes: {post['likes']}\n")
```

### 13.3.2 Social Media Sentiment: Reddit, Twitter/X, StockTwits

```python
import praw  # Reddit API
from typing import List, Dict
import pandas as pd

class RedditFinancialSentiment:
    """
    Extract sentiment from Reddit financial subreddits.
    
    Key subreddits for Indian stocks:
    - r/stocks
    - r/NSE (if exists)
    - r/IndianStocks
    - r/wallstreetbets
    
    Setup:
    1. Install: pip install praw
    2. Create Reddit app: https://www.reddit.com/prefs/apps
    3. Get credentials: client_id, client_secret, user_agent
    """
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit API client.
        
        Args:
            client_id: Reddit app client ID
            client_secret: Reddit app client secret
            user_agent: Identifier (e.g., "FinancialAnalysis/1.0")
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def search_posts(
        self,
        ticker: str,
        subreddit: str = "stocks",
        limit: int = 100
    ) -> List[Dict]:
        """
        Search for posts about a ticker.
        
        Args:
            ticker: Stock ticker to search
            subreddit: Subreddit to search
            limit: Number of posts
        
        Returns:
            List of posts with metadata
        """
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []
            
            for submission in sub.search(ticker, time_filter='month', limit=limit):
                posts.append({
                    'title': submission.title,
                    'body': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'created_utc': submission.created_utc,
                    'url': submission.url
                })
            
            return posts
        
        except Exception as e:
            print(f"Error searching Reddit: {e}")
            return []
    
    def aggregate_sentiment(self, posts: List[Dict]) -> Dict[str, float]:
        """
        Aggregate sentiment from posts.
        
        Args:
            posts: List of posts
        
        Returns:
            Aggregated metrics
        """
        if not posts:
            return {
                'avg_score': 0.0,
                'total_engagement': 0,
                'post_count': 0,
                'avg_comments': 0.0
            }
        
        scores = [p['score'] for p in posts]
        comments = [p['num_comments'] for p in posts]
        
        return {
            'avg_score': float(pd.Series(scores).mean()),
            'total_engagement': int(sum(scores) + sum(comments)),
            'post_count': len(posts),
            'avg_comments': float(pd.Series(comments).mean()),
            'median_score': float(pd.Series(scores).median())
        }


class TwitterFinancialSentiment:
    """
    Extract sentiment from Twitter/X using tweepy.
    
    Note: Twitter API v2 requires paid tier for data.
    Alternatives:
    - Use PRAW for Reddit (free)
    - Use StockTwits API (free tier available)
    - Save tweets for analysis (older tweets, less current)
    
    Setup:
    1. Apply for Twitter Developer account
    2. Get API keys
    3. Install: pip install tweepy
    """
    
    def __init__(self, bearer_token: str = None):
        """
        Initialize Twitter API client.
        
        Args:
            bearer_token: Twitter API v2 bearer token
        """
        self.bearer_token = bearer_token
        # Would need tweepy initialization here
        self.client = None
    
    def search_tweets(
        self,
        ticker: str,
        query_keywords: List[str] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search for tweets about a ticker.
        
        Args:
            ticker: Stock ticker
            query_keywords: Additional keywords to search
            max_results: Number of tweets
        
        Returns:
            List of tweets
        """
        if not self.client or not self.bearer_token:
            print("Twitter API not configured")
            return []
        
        # Would implement tweet search here
        # For now, return empty (requires paid API)
        return []
```

### 13.3.3 Google Trends as a Signal

```python
import pandas as pd
from pytrends.request import TrendReq
from typing import Dict, List
from datetime import datetime, timedelta

class GoogleTrendsAnalyzer:
    """
    Use Google Trends data as a sentiment/interest signal.
    
    Key insight (Andrade et al., 2013):
    High search volume for a stock predicts higher short-term volatility
    and positive returns (retail investor attention).
    
    Interpretation:
    - Rising search volume → increased retail interest
    - Correlates with stock price increases (short-term)
    - More predictive for smaller stocks (less efficient)
    
    Setup:
    pip install pytrends
    """
    
    def __init__(self):
        """Initialize Google Trends client."""
        self.pytrends = TrendReq(hl='en-US', tz=330)  # 330 min = IST
    
    def get_trending_searches(self, ticker: str, timeframe: str = 'today 1-m') -> pd.DataFrame:
        """
        Get Google Trends interest over time.
        
        Args:
            ticker: Stock ticker to search (e.g., 'INFY')
            timeframe: Period ('today 1-m' = last 1 month, 'today 3-m' = 3 months, etc.)
        
        Returns:
            DataFrame with daily interest level (0-100 scale)
        """
        try:
            self.pytrends.build_payload([ticker], timeframe=timeframe)
            data = self.pytrends.interest_over_time()
            return data
        
        except Exception as e:
            print(f"Error fetching Google Trends: {e}")
            return pd.DataFrame()
    
    def compare_trends(self, tickers: List[str], timeframe: str = 'today 1-m') -> pd.DataFrame:
        """
        Compare search interest between multiple tickers.
        
        Args:
            tickers: List of tickers to compare
            timeframe: Time period
        
        Returns:
            DataFrame with comparative interest (0-100)
        """
        try:
            self.pytrends.build_payload(tickers, timeframe=timeframe)
            data = self.pytrends.interest_over_time()
            return data
        
        except Exception as e:
            print(f"Error comparing trends: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get queries related to a ticker.
        
        Returns:
            {
                'top': top related queries,
                'rising': fastest-rising queries
            }
        """
        try:
            self.pytrends.build_payload([ticker])
            queries = self.pytrends.related_queries()
            
            return {
                'top': queries[ticker]['top'],
                'rising': queries[ticker]['rising']
            }
        
        except Exception as e:
            print(f"Error getting related queries: {e}")
            return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}


# Example
gt_analyzer = GoogleTrendsAnalyzer()

# Get 3-month trends for INFY
print("Fetching 3-month Google Trends for INFY...")
trends = gt_analyzer.get_trending_searches('INFY', timeframe='today 3-m')

if not trends.empty:
    print(f"Interest over time (last 10 days):")
    print(trends.tail(10))
    
    # Calculate trend momentum
    recent_interest = trends['INFY'].iloc[-7:].mean()
    older_interest = trends['INFY'].iloc[-30:-7].mean()
    momentum = (recent_interest - older_interest) / max(older_interest, 1)
    
    print(f"\nTrend Momentum (recent vs older): {momentum:+.2%}")
    if momentum > 0.1:
        print("Increasing search interest - Bullish signal")
    elif momentum < -0.1:
        print("Decreasing search interest - Bearish signal")
```

### 13.3.4 Other Alternative Data Sources (Conceptual Overview)

```python
"""
OTHER ALTERNATIVE DATA SOURCES FOR TRADING SIGNALS

This section provides a conceptual overview and code structure for
integrating additional alternative data sources.

1. SATELLITE AND GEOSPATIAL DATA
   - Company parking lots (activity level)
   - Store foot traffic (retail sales proxy)
   - Port activity (trade flows)
   - Construction activity (economic activity)
   
   Providers:
   - Orbital Insight
   - Descartes Labs
   - Maxar Technologies
   
   Implementation:
   Usually proprietary APIs requiring contracts. Not suitable for retail traders
   but worth understanding the concept.
   
   Example: Retailer foot traffic signal
   
   import numpy as np
   
   def correlate_foot_traffic_stock(foot_traffic: np.ndarray, 
                                     stock_returns: np.ndarray) -> float:
       '''
       Measure correlation between foot traffic and stock returns.
       
       Higher correlation indicates foot traffic as predictive signal.
       '''
       return np.corrcoef(foot_traffic, stock_returns)[0, 1]


2. JOB POSTINGS AND EMPLOYEE REVIEW DATA
   - Growing job postings → hiring → growth signal
   - Declining job postings → contraction
   - Employee reviews (Glassdoor) → sentiment about company culture
   
   Providers:
   - Burning Glass Technologies
   - LinkedIn data (if you have access)
   - Glassdoor reviews (scrapeable, check ToS)
   
   Implementation:
"""

class JobPostingAnalyzer:
    """
    Analyze job posting trends as hiring/growth signal.
    
    Hypothesis:
    Companies hiring rapidly → expect growth
    Companies reducing headcount → expect contraction
    
    Data source: LinkedIn, Dice, job boards (if scrapeable)
    """
    
    def analyze_job_growth(self, postings_today: int, postings_30d_ago: int) -> Dict[str, float]:
        """
        Analyze job posting growth.
        
        Args:
            postings_today: Current number of job postings
            postings_30d_ago: Number 30 days ago
        
        Returns:
            Growth metric and interpretation
        """
        if postings_30d_ago == 0:
            growth_rate = 1.0 if postings_today > 0 else 0.0
        else:
            growth_rate = (postings_today - postings_30d_ago) / postings_30d_ago
        
        return {
            'hiring_growth_rate': float(growth_rate),
            'interpretation': self._interpret_hiring(growth_rate)
        }
    
    @staticmethod
    def _interpret_hiring(growth_rate: float) -> str:
        """Interpret hiring growth rate."""
        if growth_rate > 0.2:
            return "Rapid hiring - Strong growth signal"
        elif growth_rate > 0.05:
            return "Moderate hiring - Positive signal"
        elif growth_rate > -0.05:
            return "Stable hiring - Neutral"
        elif growth_rate > -0.2:
            return "Slight contraction - Weak negative signal"
        else:
            return "Significant headcount reduction - Strong negative signal"


class EmployeeReviewAnalyzer:
    """
    Analyze employee reviews (Glassdoor, AmbitionBox).
    
    Hypothesis:
    High employee satisfaction → lower turnover → stable business
    Low satisfaction → turnover → disruption
    """
    
    def aggregate_sentiment(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Aggregate employee review sentiment.
        
        Args:
            reviews: List of reviews with 'rating' and 'text'
        
        Returns:
            Sentiment metrics
        """
        if not reviews:
            return {'avg_rating': 0.0, 'review_count': 0}
        
        ratings = [r.get('rating', 3) for r in reviews]
        
        return {
            'avg_rating': float(np.mean(ratings)),
            'review_count': len(reviews),
            'trend': 'improving' if ratings[-5:] > ratings[-10:-5] else 'declining'
        }


"""
3. SUPPLY CHAIN DATA
   - Shipping container prices (global trade volume)
   - Freight rates (economic activity)
   - Port congestion (trade flows)
   - Commodity prices (input costs)
   
   These affect profit margins and are often leading indicators.

4. CREDIT CARD AND CONSUMER SPENDING DATA
   - Visa, MasterCard transaction data
   - PayPal, Square payment volumes
   - Amazon seller data
   
   These are real-time indicators of consumer spending,
   more timely than official retail sales data.

5. EMAIL AND SEARCH VOLUME AGGREGATORS
   - Email traffic patterns (business activity)
   - Search advertising costs (competition intensity)
   - Domain registration activity (new business formation)

6. WEATHER AND NATURAL DISASTERS
   - Unusual weather → agricultural impact
   - Natural disasters → insurance/reconstruction demand
   - Seasonal patterns → retail/tourism impact

IMPLEMENTATION STRATEGY FOR NSE TRADING:

For most retail traders in India:
1. Focus on FREELY AVAILABLE data: Google Trends, StockTwits API, Reddit
2. Avoid expensive alternative data (satellite imagery, etc.)
3. Combine with traditional technical + fundamental analysis
4. Backtest heavily before deploying signals
5. Monitor for overfitting to past data
"""
```

### 13.3.5 Integrating Alternative Data into Zerodha Trading System

```python
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Assume we have kiteconnect from Zerodha available
# from kiteconnect import KiteConnect

class AlternativeDataSignalProvider:
    """
    Integrate all alternative data sources into a unified signal system.
    
    This class combines:
    - News sentiment (FinBERT)
    - Earnings tone analysis
    - Google Trends
    - Social media sentiment
    - Other signals
    
    Into a single composite trading signal.
    """
    
    def __init__(self, stock_symbols: List[str]):
        """
        Initialize signal provider.
        
        Args:
            stock_symbols: List of NSE stock symbols (e.g., ['INFY', 'TCS'])
        """
        self.symbols = stock_symbols
        self.logger = logging.getLogger(__name__)
        
        # Initialize component analyzers
        self.finbert = FinBERTSentiment(device='cpu')
        self.tone_analyzer = ManagementToneAnalyzer()
        self.gt_analyzer = GoogleTrendsAnalyzer()
        self.news_signal_gen = NewsSignalGenerator()
        
        # Signal weights (can be optimized via backtesting)
        self.signal_weights = {
            'news_sentiment': 0.4,
            'earnings_tone': 0.2,
            'google_trends': 0.2,
            'social_sentiment': 0.2
        }
    
    def compute_news_sentiment_signal(self, articles: List[str]) -> float:
        """
        Compute news sentiment signal using FinBERT.
        
        Args:
            articles: List of news article texts
        
        Returns:
            Sentiment signal in [-1, 1]
        """
        if not articles:
            return 0.0
        
        sentiments = self.finbert.analyze_batch(articles)
        sentiment_scores = [s['sentiment'] for s in sentiments]
        
        # Average sentiment
        return float(np.mean(sentiment_scores))
    
    def compute_earnings_tone_signal(self, transcript: str) -> float:
        """
        Compute earnings tone signal.
        
        Args:
            transcript: Earnings call transcript
        
        Returns:
            Tone signal in [-1, 1]
        """
        tone_results = self.tone_analyzer.analyze_tone(transcript)
        
        # Combine confidence and bullish scores
        signal = (tone_results['confidence_score'] + tone_results['bullish_score']) / 2
        
        return float(np.clip(signal, -1, 1))
    
    def compute_google_trends_signal(self, ticker: str) -> float:
        """
        Compute Google Trends signal.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Trend momentum signal
        """
        try:
            trends = self.gt_analyzer.get_trending_searches(ticker, timeframe='today 1-m')
            
            if trends.empty or len(trends) < 7:
                return 0.0
            
            # Calculate 7-day vs 30-day momentum
            recent = trends[ticker].iloc[-7:].mean()
            older = trends[ticker].iloc[-30:-7].mean()
            
            if older == 0:
                momentum = 0.0
            else:
                momentum = (recent - older) / older
            
            # Normalize to [-1, 1]
            return float(np.clip(momentum / 0.5, -1, 1))
        
        except Exception as e:
            self.logger.warning(f"Error computing Google Trends signal: {e}")
            return 0.0
    
    def compute_composite_signal(
        self,
        ticker: str,
        articles: List[str] = None,
        earnings_transcript: str = None
    ) -> Dict[str, float]:
        """
        Compute composite alternative data signal.
        
        Args:
            ticker: Stock ticker
            articles: News articles (optional)
            earnings_transcript: Earnings transcript (optional)
        
        Returns:
            {
                'news_signal': float,
                'earnings_signal': float,
                'trends_signal': float,
                'composite_signal': float,
                'confidence': float
            }
        """
        signals = {}
        
        # News sentiment
        if articles:
            signals['news_signal'] = self.compute_news_sentiment_signal(articles)
        else:
            signals['news_signal'] = 0.0
        
        # Earnings tone
        if earnings_transcript:
            signals['earnings_signal'] = self.compute_earnings_tone_signal(earnings_transcript)
        else:
            signals['earnings_signal'] = 0.0
        
        # Google Trends
        signals['trends_signal'] = self.compute_google_trends_signal(ticker)
        
        # Composite signal (weighted average)
        weights_used = {}
        composite = 0.0
        total_weight = 0.0
        
        for signal_type, weight in self.signal_weights.items():
            signal_key = signal_type.replace('_', '_signal')
            if signal_type not in ['social_sentiment']:  # Skip if not available
                value = signals.get(signal_key, 0.0)
                if value != 0.0:  # Only weight non-zero signals
                    composite += value * weight
                    total_weight += weight
        
        if total_weight > 0:
            composite = composite / total_weight
        
        # Confidence (average absolute value of inputs)
        confidence = np.mean([abs(v) for v in signals.values()])
        
        return {
            'news_signal': float(signals['news_signal']),
            'earnings_signal': float(signals['earnings_signal']),
            'trends_signal': float(signals['trends_signal']),
            'composite_signal': float(np.clip(composite, -1, 1)),
            'confidence': float(confidence)
        }
    
    def generate_trading_recommendation(
        self,
        ticker: str,
        composite_signal: float,
        confidence: float,
        signal_threshold: float = 0.3
    ) -> Dict[str, str]:
        """
        Convert composite signal to trading recommendation.
        
        Args:
            ticker: Stock ticker
            composite_signal: Composite signal in [-1, 1]
            confidence: Confidence level in [0, 1]
            signal_threshold: Threshold for action
        
        Returns:
            {
                'action': 'BUY' / 'SELL' / 'HOLD',
                'reasoning': str,
                'position_size': 'SMALL' / 'MEDIUM' / 'LARGE',
                'stop_loss_level': float (if applicable)
            }
        """
        if confidence < 0.2:
            return {
                'action': 'HOLD',
                'reasoning': 'Low confidence in alternative data signals',
                'position_size': 'NONE'
            }
        
        if composite_signal > signal_threshold:
            return {
                'action': 'BUY',
                'reasoning': f'Positive alternative data signals (score: {composite_signal:.2f})',
                'position_size': 'LARGE' if confidence > 0.6 else 'MEDIUM'
            }
        
        elif composite_signal < -signal_threshold:
            return {
                'action': 'SELL',
                'reasoning': f'Negative alternative data signals (score: {composite_signal:.2f})',
                'position_size': 'LARGE' if confidence > 0.6 else 'MEDIUM'
            }
        
        else:
            return {
                'action': 'HOLD',
                'reasoning': 'Alternative data signals are mixed/neutral',
                'position_size': 'NONE'
            }


# Example usage
alt_data_provider = AlternativeDataSignalProvider(['INFY', 'TCS', 'RELIANCE'])

# Simulated data
sample_articles = [
    "Infosys reported strong Q3 earnings with 15% growth",
    "IT sector showing resilience amid global recession fears",
    "Digital transformation spending driving Infosys growth"
]

sample_earnings = """
We're confident about our market position and expect significant growth.
Digital services showed strong momentum with positive guidance.
"""

# Compute signals
signals = alt_data_provider.compute_composite_signal(
    'INFY',
    articles=sample_articles,
    earnings_transcript=sample_earnings
)

print(f"INFY Alternative Data Signals:")
print(f"  News Sentiment: {signals['news_signal']:+.3f}")
print(f"  Earnings Tone: {signals['earnings_signal']:+.3f}")
print(f"  Google Trends: {signals['trends_signal']:+.3f}")
print(f"  Composite Signal: {signals['composite_signal']:+.3f}")
print(f"  Confidence: {signals['confidence']:.3f}\n")

recommendation = alt_data_provider.generate_trading_recommendation(
    'INFY',
    signals['composite_signal'],
    signals['confidence'],
    signal_threshold=0.2
)

print(f"Trading Recommendation: {recommendation['action']}")
print(f"Position Size: {recommendation['position_size']}")
print(f"Reasoning: {recommendation['reasoning']}")
```

---

## Chapter Summary and Practical Integration

### Key Takeaways

1. **Alternative data contains predictive information** that traditional data (prices, financials) doesn't capture. Markets take time to incorporate this information.

2. **NLP and transformers are powerful tools** for financial text analysis. FinBERT specifically is trained on financial language and captures context far better than dictionary methods.

3. **Signal generation requires careful methodology**:
   - Aggregate article-level to daily/stock level
   - Normalize against baselines (z-score)
   - Combine multiple sources (news, earnings, trends, social)
   - Always backtest before trading

4. **Legal and ethical considerations matter**. Web scraping can violate Terms of Service. Use official APIs when available.

5. **For NSE trading**: Focus on freely available sources (Google Trends, StockTwits, Reddit) combined with your existing Zerodha API integration.

### Integration with Your Zerodha System

```python
"""
PRACTICAL EXAMPLE: Integrating alternative data into Zerodha trading

This would be added to your main trading system (see Chapter 11)
"""

class ZerodhaAlternativeDataTrader:
    """
    Extended Zerodha trader with alternative data signals.
    """
    
    def __init__(self, kite_connection):
        """
        Initialize trader.
        
        Args:
            kite_connection: Active KiteConnect instance
        """
        self.kite = kite_connection
        self.alt_data_provider = AlternativeDataSignalProvider(['INFY', 'TCS'])
    
    def place_trade_with_alt_signals(
        self,
        symbol: str,
        articles: List[str],
        earnings_transcript: str = None
    ):
        """
        Place trade based on alternative data signals.
        
        Args:
            symbol: Stock symbol
            articles: Recent news articles
            earnings_transcript: Latest earnings call (if available)
        """
        # Compute alternative data signals
        signals = self.alt_data_provider.compute_composite_signal(
            symbol,
            articles=articles,
            earnings_transcript=earnings_transcript
        )
        
        # Get recommendation
        recommendation = self.alt_data_provider.generate_trading_recommendation(
            symbol,
            signals['composite_signal'],
            signals['confidence'],
            signal_threshold=0.25
        )
        
        # Only trade if high confidence
        if signals['confidence'] < 0.3:
            print(f"Skipping {symbol}: Low signal confidence")
            return
        
        # Place order based on recommendation
        if recommendation['action'] == 'BUY':
            quantity = self._get_quantity(symbol, recommendation['position_size'])
            
            order_id = self.kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=symbol,
                transaction_type='BUY',
                quantity=quantity,
                price=0,
                order_type='MARKET',
                tag=f"alt_data_{signals['composite_signal']:.2f}"
            )
            
            print(f"BUY order placed: {symbol} x {quantity} "
                  f"(Signal: {signals['composite_signal']:.2f})")
        
        elif recommendation['action'] == 'SELL':
            quantity = self._get_quantity(symbol, recommendation['position_size'])
            
            order_id = self.kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=symbol,
                transaction_type='SELL',
                quantity=quantity,
                price=0,
                order_type='MARKET',
                tag=f"alt_data_{signals['composite_signal']:.2f}"
            )
            
            print(f"SELL order placed: {symbol} x {quantity} "
                  f"(Signal: {signals['composite_signal']:.2f})")
    
    def _get_quantity(self, symbol: str, position_size: str) -> int:
        """Determine order quantity based on position size."""
        quote = self.kite.quote(f"NSE:{symbol}")
        current_price = quote[f"NSE:{symbol}"]['last_price']
        
        # Allocate capital based on position size
        if position_size == 'LARGE':
            capital = 50000  # Rs. 50,000
        elif position_size == 'MEDIUM':
            capital = 25000  # Rs. 25,000
        else:
            capital = 10000  # Rs. 10,000
        
        return int(capital / current_price)
```

---

## References and Further Reading

1. **NLP and Sentiment**:
   - Loughran, T., & McDonald, B. (2011). "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks." Journal of Finance.
   - Huang, A. H., et al. (2014). "Analyst Information Discovery and the Profitability of Momentum Trading." Review of Finance.

2. **Event Studies**:
   - Fama, E. F. (1969). "Efficient capital markets: A review of theory and empirical work." Journal of Finance.
   - MacKinlay, A. C. (1997). "Event Studies in Economics and Finance." Journal of Economic Literature.

3. **Alternative Data**:
   - Hou, K., et al. (2015). "Digesting Anomalies: An Investment Approach." Journal of Financial Economics.
   - Tetlock, P. C. (2007). "Giving Content to Investor Sentiment: The Role of Media in Stock Price Discovery." Journal of Finance.

4. **Earnings Quality and Obfuscation**:
   - Larcker, D. F., & Zakolyukina, A. A. (2012). "Detecting Deceptive Discussions in Conferences Calls." Journal of Accounting Research.

5. **Python Libraries**:
   - HuggingFace Transformers: https://huggingface.co/transformers/
   - PyTrends: https://github.com/GeneralMills/pytrends
   - PRAW (Reddit API): https://praw.readthedocs.io/

---

## Chapter 13 Complete

You now have:

- ✓ Production-grade FinBERT sentiment pipeline
- ✓ News signal aggregation system
- ✓ Earnings analysis (tone, readability, changes)
- ✓ Alternative data sources (Google Trends, Reddit, web scraping)
- ✓ Composite signal generation
- ✓ Integration with Zerodha trading system

**Next steps**: Backtest these signals on historical NSE data (see Chapter 12 for backtesting framework). Combine with traditional technical/fundamental signals for robust trading systems.

Remember: **Alternative data works because humans take time to process information. Automation extracts value from that lag.**
