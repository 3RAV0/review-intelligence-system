"""
Preprocessing pipeline for Amazon Reviews dataset.

This module provides a reusable class for preprocessing review data:
duplicate removal, text cleaning, sentiment label engineering,
and stratified train/val/test splitting.
"""

import re
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


class ReviewPreprocessor:
    """
    Preprocessing pipeline for Amazon multi-category reviews.
    
    Usage:
        preprocessor = ReviewPreprocessor()
        train_df, val_df, test_df = preprocessor.fit_transform(df)
    """
    
    def __init__(
        self,
        min_word_count: int = 5,
        test_size: float = 0.20,
        val_size: float = 0.50,
        random_seed: int = 42
    ):
        """
        Args:
            min_word_count: Minimum words required per review
            test_size: Fraction of data for temp set (val + test combined)
            val_size: Fraction of temp set assigned to val (rest is test)
            random_seed: Reproducibility seed
        """
        self.min_word_count = min_word_count
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        
        # Tracking
        self.loss_log: List[Dict] = []
        self.text_duplicates: pd.DataFrame = pd.DataFrame()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Light text cleaning suitable for transformer models."""
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'http\S+|www\.\S+', ' ', text)  # URLs
        text = re.sub(r'<[^>]+>', ' ', text)  # HTML tags
        text = re.sub(r'\s+', ' ', text)  # Whitespace
        
        return text.strip()
    
    @staticmethod
    def rating_to_sentiment(rating: float) -> str:
        """Convert 1-5 rating to 3-class sentiment."""
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    def _log(self, step: str, df_current: pd.DataFrame, df_previous: pd.DataFrame = None):
        """Track row counts at each step."""
        if df_previous is not None:
            removed = len(df_previous) - len(df_current)
            pct = (removed / len(df_previous) * 100) if len(df_previous) > 0 else 0
        else:
            removed, pct = 0, 0
        
        self.loss_log.append({
            'step': step,
            'rows_remaining': len(df_current),
            'rows_removed': removed,
            'pct_removed': round(pct, 2)
        })
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact and text-level duplicates. Save text duplicates separately."""
        hashable_cols = [
            'rating', 'title', 'text', 'asin', 'parent_asin',
            'user_id', 'timestamp', 'helpful_vote',
            'verified_purchase', 'category'
        ]
        
        # Exact duplicates
        df_before = df.copy()
        df = df.drop_duplicates(subset=hashable_cols, keep='first').reset_index(drop=True)
        self._log("drop_exact_duplicates", df, df_before)
        
        # Save text duplicates (for fake review detection module)
        self.text_duplicates = df[df.duplicated(subset=['text'], keep='first')].copy()
        
        # Drop text duplicates
        df_before = df.copy()
        df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        self._log("drop_text_duplicates", df, df_before)
        
        return df
    
    def filter_short_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty and very short reviews."""
        df_before = df.copy()
        
        df = df.dropna(subset=['text']).reset_index(drop=True)
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'] != ''].reset_index(drop=True)
        df = df[df['text'].str.split().str.len() >= self.min_word_count].reset_index(drop=True)
        
        self._log("filter_short_reviews", df, df_before)
        return df
    
    def apply_cleaning_and_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning and create sentiment labels."""
        # Clean text
        df['text_clean'] = df['text'].apply(self.clean_text)
        df['word_count_clean'] = df['text_clean'].str.split().str.len()
        
        # Filter again (cleaning might shorten text)
        df_before = df.copy()
        df = df[df['word_count_clean'] >= self.min_word_count].reset_index(drop=True)
        self._log("post_cleaning_filter", df, df_before)
        
        # Create sentiment label
        df['sentiment'] = df['rating'].apply(self.rating_to_sentiment)
        
        return df
    
    def split(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified split on (sentiment, category) combined key."""
        df['strat_key'] = df['sentiment'] + '_' + df['category']
        
        train_df, temp_df = train_test_split(
            df, test_size=self.test_size,
            random_state=self.random_seed,
            stratify=df['strat_key']
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=self.val_size,
            random_state=self.random_seed,
            stratify=temp_df['strat_key']
        )
        
        # Cleanup
        for split_df in [train_df, val_df, test_df]:
            split_df.drop(columns=['strat_key'], inplace=True)
        
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        )
    
    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run full preprocessing pipeline and return (train, val, test) splits."""
        self.loss_log = []
        self._log("initial", df)
        
        df = self.remove_duplicates(df)
        df = self.filter_short_reviews(df)
        df = self.apply_cleaning_and_labeling(df)
        
        return self.split(df)
    
    def get_loss_summary(self) -> pd.DataFrame:
        """Return preprocessing data loss summary."""
        return pd.DataFrame(self.loss_log)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "../data/processed"
) -> None:
    """Save splits as parquet files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_path / "train.parquet", index=False)
    val_df.to_parquet(output_path / "val.parquet", index=False)
    test_df.to_parquet(output_path / "test.parquet", index=False)
    
    print(f"Splits saved to {output_path}")