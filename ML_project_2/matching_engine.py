"""
Property Matching Engine
=========================
Loads user preferences and property listings, and scores every user-property
pair using a hybrid of weighted numerical matching and TF-IDF semantic
similarity.
"""

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


class PropertyMatchingSystem:
    """
    Main class for property matching system that handles data processing,
    match score calculation, and visualization.
    """

    def __init__(self, data_file='Case Study 2 Data.xlsx'):
        """Initialize the system and load data."""
        self.data_file = data_file
        self.user_df = None
        self.prop_df = None
        self.user_df_processed = None
        self.prop_df_processed = None
        self.match_scores = None
        self.scaler = MinMaxScaler()

    def load_data(self):
        """Load user preferences and property data from Excel file."""
        print("Loading data...")
        self.user_df = pd.read_excel(self.data_file, sheet_name='User Data')
        self.prop_df = pd.read_excel(self.data_file, sheet_name='Property Data')

        print(f"Loaded {len(self.user_df)} user preferences")
        print(f"Loaded {len(self.prop_df)} properties")
        return self

    def parse_price(self, price_str):
        """Convert price string (e.g., '$500k') to numeric value."""
        if pd.isna(price_str):
            return np.nan

        # Remove $ and other characters, keep only numbers and k/m
        price_str = str(price_str).strip().lower()
        price_str = re.sub(r'[^\d.km]', '', price_str)

        # Convert k (thousands) and m (millions) to actual numbers
        if 'k' in price_str:
            return float(price_str.replace('k', '')) * 1000
        elif 'm' in price_str:
            return float(price_str.replace('m', '')) * 1000000
        else:
            return float(price_str)

    def preprocess_data(self):
        """Clean and preprocess both datasets."""
        print("\nPreprocessing data...")

        # Process user preferences
        self.user_df_processed = self.user_df.copy()
        self.user_df_processed['Budget_Numeric'] = self.user_df_processed['Budget'].apply(self.parse_price)

        # Process property data
        self.prop_df_processed = self.prop_df.copy()
        self.prop_df_processed['Price_Numeric'] = self.prop_df_processed['Price'].apply(self.parse_price)

        # Normalize numerical features for fair comparison
        self.user_df_processed['Budget_Norm'] = self.scaler.fit_transform(
            self.user_df_processed[['Budget_Numeric']]
        )
        self.user_df_processed['Bedrooms_Norm'] = self.scaler.fit_transform(
            self.user_df_processed[['Bedrooms']]
        )
        self.user_df_processed['Bathrooms_Norm'] = self.scaler.fit_transform(
            self.user_df_processed[['Bathrooms']]
        )

        self.prop_df_processed['Price_Norm'] = self.scaler.fit_transform(
            self.prop_df_processed[['Price_Numeric']]
        )
        self.prop_df_processed['Bedrooms_Norm'] = self.scaler.fit_transform(
            self.prop_df_processed[['Bedrooms']]
        )
        self.prop_df_processed['Bathrooms_Norm'] = self.scaler.fit_transform(
            self.prop_df_processed[['Bathrooms']]
        )
        self.prop_df_processed['LivingArea_Norm'] = self.scaler.fit_transform(
            self.prop_df_processed[['Living Area (sq ft)']]
        )

        print("Data preprocessing complete!")
        return self

    def calculate_weighted_score(self, user_row, prop_row):
        """
        Calculate weighted match score based on numerical features.

        Methodology:
        - Budget/Price matching: 40% weight (closer is better)
        - Bedroom matching: 25% weight (exact match preferred)
        - Bathroom matching: 20% weight (exact match preferred)
        - Living area: 15% weight (bonus for larger spaces)

        Returns score between 0-100
        """
        # Budget/Price similarity (40% weight)
        if prop_row['Price_Numeric'] <= user_row['Budget_Numeric']:
            price_score = 1 - abs(prop_row['Price_Numeric'] - user_row['Budget_Numeric']) / user_row['Budget_Numeric']
        else:
            # Heavy penalty if over budget
            price_score = max(0, 1 - 2 * (prop_row['Price_Numeric'] - user_row['Budget_Numeric']) / user_row['Budget_Numeric'])

        # Bedroom matching (25% weight)
        bedroom_diff = abs(prop_row['Bedrooms'] - user_row['Bedrooms'])
        bedroom_score = max(0, 1 - bedroom_diff * 0.25)

        # Bathroom matching (20% weight)
        bathroom_diff = abs(prop_row['Bathrooms'] - user_row['Bathrooms'])
        bathroom_score = max(0, 1 - bathroom_diff * 0.3)

        # Living area bonus (15% weight)
        area_score = min(1.0, prop_row['LivingArea_Norm'])

        weighted_score = (
            price_score * 0.40 +
            bedroom_score * 0.25 +
            bathroom_score * 0.20 +
            area_score * 0.15
        )

        return weighted_score * 100

    def calculate_semantic_similarity(self, user_text, prop_text):
        """
        Calculate semantic similarity between user preferences and property
        descriptions using TF-IDF and cosine similarity.
        """
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            vectors = vectorizer.fit_transform([user_text, prop_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity * 100
        except Exception:
            return 0.0

    def calculate_hybrid_score(self, user_row, prop_row):
        """
        Hybrid approach combining weighted numerical matching and semantic
        similarity.

        Final score = 70% weighted numerical + 30% semantic similarity
        """
        weighted = self.calculate_weighted_score(user_row, prop_row)
        semantic = self.calculate_semantic_similarity(
            user_row['Qualitative Description'],
            prop_row['Qualitative Description']
        )

        hybrid = weighted * 0.70 + semantic * 0.30
        return hybrid, weighted, semantic

    def calculate_all_matches(self):
        """
        Calculate match scores for all user-property pairs.
        Returns a detailed dataframe with all scores.
        """
        print("\nCalculating match scores for all user-property pairs...")

        results = []

        for _, user in self.user_df_processed.iterrows():
            for _, prop in self.prop_df_processed.iterrows():
                hybrid, weighted, semantic = self.calculate_hybrid_score(user, prop)

                results.append({
                    'User_ID': user['User ID'],
                    'Property_ID': prop['Property ID'],
                    'Match_Score': hybrid,
                    'Weighted_Score': weighted,
                    'Semantic_Score': semantic,
                    'User_Budget': user['Budget'],
                    'Property_Price': prop['Price'],
                    'User_Bedrooms': user['Bedrooms'],
                    'Property_Bedrooms': prop['Bedrooms'],
                    'User_Bathrooms': user['Bathrooms'],
                    'Property_Bathrooms': prop['Bathrooms'],
                    'Property_Area': prop['Living Area (sq ft)'],
                    'User_Description': user['Qualitative Description'][:100] + '...',
                    'Property_Description': prop['Qualitative Description'][:100] + '...'
                })

        self.match_scores = pd.DataFrame(results)
        print(f"Calculated {len(self.match_scores)} match scores!")

        return self.match_scores

    def get_top_matches(self, user_id, top_n=5):
        """Get top N property matches for a specific user."""
        user_matches = self.match_scores[self.match_scores['User_ID'] == user_id]
        top_matches = user_matches.nlargest(top_n, 'Match_Score')
        return top_matches
