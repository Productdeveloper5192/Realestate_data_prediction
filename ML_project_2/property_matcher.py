"""
Property Matching System with AI/ML Techniques
===============================================
This comprehensive solution matches properties with user preferences using multiple
AI/ML techniques including weighted scoring, cosine similarity, and semantic matching.

Author: Agent Mira Data Science Case Study
Date: January 2026

Key Features:
- Data preprocessing and normalization
- Multiple matching algorithms (weighted, cosine similarity, semantic)
- Comprehensive visualizations
- Interactive Streamlit UI for real-time matching
- Detailed match score explanations

Usage:
    # Run analysis and generate visualizations
    python property_matcher.py

    # Run interactive UI
    streamlit run property_matcher.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import streamlit for UI (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available. Install with: pip install streamlit")


# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

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
        # We'll create normalized versions for matching while keeping originals for display
        user_numeric = self.user_df_processed[['Budget_Numeric', 'Bedrooms', 'Bathrooms']].values
        prop_numeric = self.prop_df_processed[['Price_Numeric', 'Bedrooms', 'Bathrooms', 'Living Area (sq ft)']].values
        
        # Store normalized features
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
    
    # ============================================================================
    # SECTION 2: MATCH SCORE CALCULATION ALGORITHMS
    # ============================================================================
    
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
        # Use inverse of percentage difference, capped at budget
        if prop_row['Price_Numeric'] <= user_row['Budget_Numeric']:
            price_score = 1 - abs(prop_row['Price_Numeric'] - user_row['Budget_Numeric']) / user_row['Budget_Numeric']
        else:
            # Heavy penalty if over budget
            price_score = max(0, 1 - 2 * (prop_row['Price_Numeric'] - user_row['Budget_Numeric']) / user_row['Budget_Numeric'])
        
        # Bedroom matching (25% weight)
        bedroom_diff = abs(prop_row['Bedrooms'] - user_row['Bedrooms'])
        bedroom_score = max(0, 1 - bedroom_diff * 0.25)  # -25% per bedroom difference
        
        # Bathroom matching (20% weight)
        bathroom_diff = abs(prop_row['Bathrooms'] - user_row['Bathrooms'])
        bathroom_score = max(0, 1 - bathroom_diff * 0.3)  # -30% per bathroom difference
        
        # Living area bonus (15% weight)
        # Normalize to 0-1 scale based on dataset
        area_score = min(1.0, prop_row['LivingArea_Norm'])
        
        # Weighted combination
        weighted_score = (
            price_score * 0.40 +
            bedroom_score * 0.25 +
            bathroom_score * 0.20 +
            area_score * 0.15
        )
        
        return weighted_score * 100  # Convert to 0-100 scale
    
    def calculate_semantic_similarity(self, user_text, prop_text):
        """
        Calculate semantic similarity between user preferences and property descriptions
        using TF-IDF and cosine similarity.
        
        This captures the textual/qualitative aspects of the matching.
        """
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        try:
            # Fit on both texts
            vectors = vectorizer.fit_transform([user_text, prop_text])
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity * 100  # Convert to 0-100 scale
        except:
            return 0.0
    
    def calculate_hybrid_score(self, user_row, prop_row):
        """
        Hybrid approach combining weighted numerical matching and semantic similarity.
        
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
    
    # ============================================================================
    # SECTION 3: VISUALIZATION AND ANALYSIS
    # ============================================================================
    
    def create_match_heatmap(self, save_path='match_heatmap.png'):
        """Create a heatmap showing match scores for all user-property pairs."""
        print("\nGenerating match score heatmap...")
        
        # Pivot data for heatmap
        heatmap_data = self.match_scores.pivot(
            index='User_ID',
            columns='Property_ID',
            values='Match_Score'
        )
        
        # Create figure
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Match Score (0-100)'},
            linewidths=0.5
        )
        plt.title('Property Match Scores - All Users vs All Properties', fontsize=16, fontweight='bold')
        plt.xlabel('Property ID', fontsize=12)
        plt.ylabel('User ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
        plt.close()
        
    def create_top_matches_chart(self, user_id, save_path=None):
        """Create a bar chart showing top property matches for a user."""
        top_matches = self.get_top_matches(user_id, top_n=10)
        
        if save_path is None:
            save_path = f'top_matches_user_{user_id}.png'
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Overall match scores
        colors = plt.cm.RdYlGn(top_matches['Match_Score'].values / 100)
        ax1.barh(top_matches['Property_ID'].astype(str), top_matches['Match_Score'], color=colors)
        ax1.set_xlabel('Match Score', fontsize=12)
        ax1.set_ylabel('Property ID', fontsize=12)
        ax1.set_title(f'Top 10 Property Matches for User {user_id}', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.grid(axis='x', alpha=0.3)
        
        # Right plot: Score breakdown
        x = np.arange(len(top_matches))
        width = 0.35
        
        ax2.bar(x - width/2, top_matches['Weighted_Score'], width, label='Weighted (Numerical)', alpha=0.8)
        ax2.bar(x + width/2, top_matches['Semantic_Score'], width, label='Semantic (Text)', alpha=0.8)
        
        ax2.set_xlabel('Property ID', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Score Breakdown: Numerical vs Semantic Matching', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_matches['Property_ID'].astype(str))
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top matches chart saved to {save_path}")
        plt.close()
    
    def create_distribution_analysis(self, save_path='score_distribution.png'):
        """Analyze and visualize the distribution of match scores."""
        print("\nGenerating score distribution analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall match score distribution
        axes[0, 0].hist(self.match_scores['Match_Score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Match Score', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of All Match Scores', fontsize=12, fontweight='bold')
        axes[0, 0].axvline(self.match_scores['Match_Score'].mean(), color='red', linestyle='--', label=f'Mean: {self.match_scores["Match_Score"].mean():.1f}')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Average match score by user
        avg_by_user = self.match_scores.groupby('User_ID')['Match_Score'].mean().sort_values(ascending=False)
        axes[0, 1].bar(avg_by_user.index.astype(str), avg_by_user.values, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('User ID', fontsize=11)
        axes[0, 1].set_ylabel('Average Match Score', fontsize=11)
        axes[0, 1].set_title('Average Match Score by User', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Average match score by property
        avg_by_prop = self.match_scores.groupby('Property_ID')['Match_Score'].mean().sort_values(ascending=False).head(15)
        axes[1, 0].bar(avg_by_prop.index.astype(str), avg_by_prop.values, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Property ID', fontsize=11)
        axes[1, 0].set_ylabel('Average Match Score', fontsize=11)
        axes[1, 0].set_title('Top 15 Properties by Average Match Score', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Weighted vs Semantic correlation
        axes[1, 1].scatter(self.match_scores['Weighted_Score'], self.match_scores['Semantic_Score'], 
                          alpha=0.5, c=self.match_scores['Match_Score'], cmap='viridis')
        axes[1, 1].set_xlabel('Weighted Score (Numerical)', fontsize=11)
        axes[1, 1].set_ylabel('Semantic Score (Text)', fontsize=11)
        axes[1, 1].set_title('Correlation: Numerical vs Semantic Matching', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        # Add correlation coefficient
        corr = self.match_scores['Weighted_Score'].corr(self.match_scores['Semantic_Score'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution analysis saved to {save_path}")
        plt.close()
    
    def generate_summary_report(self, save_path='match_summary_report.txt'):
        """Generate a comprehensive text summary of the matching results."""
        print("\nGenerating summary report...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PROPERTY MATCHING SYSTEM - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Users: {len(self.user_df_processed)}\n")
            f.write(f"Total Properties: {len(self.prop_df_processed)}\n")
            f.write(f"Total Match Scores Calculated: {len(self.match_scores)}\n\n")
            
            f.write(f"Average Match Score: {self.match_scores['Match_Score'].mean():.2f}\n")
            f.write(f"Median Match Score: {self.match_scores['Match_Score'].median():.2f}\n")
            f.write(f"Std Dev: {self.match_scores['Match_Score'].std():.2f}\n")
            f.write(f"Min Score: {self.match_scores['Match_Score'].min():.2f}\n")
            f.write(f"Max Score: {self.match_scores['Match_Score'].max():.2f}\n\n")
            
            # Top matches for each user
            f.write("\n" + "="*80 + "\n")
            f.write("TOP 3 PROPERTY MATCHES FOR EACH USER\n")
            f.write("="*80 + "\n\n")
            
            for user_id in sorted(self.user_df_processed['User ID'].unique()):
                user_info = self.user_df_processed[self.user_df_processed['User ID'] == user_id].iloc[0]
                top_matches = self.get_top_matches(user_id, top_n=3)
                
                f.write(f"\nUSER {user_id}\n")
                f.write(f"Budget: {user_info['Budget']}, Bedrooms: {user_info['Bedrooms']}, Bathrooms: {user_info['Bathrooms']}\n")
                f.write(f"Preferences: {user_info['Qualitative Description'][:150]}...\n")
                f.write("-"*80 + "\n")
                
                for idx, match in top_matches.iterrows():
                    f.write(f"\n  Property {match['Property_ID']} - Match Score: {match['Match_Score']:.1f}\n")
                    f.write(f"  Price: {match['Property_Price']}, Bedrooms: {int(match['Property_Bedrooms'])}, ")
                    f.write(f"Bathrooms: {int(match['Property_Bathrooms'])}, Area: {int(match['Property_Area'])} sq ft\n")
                    f.write(f"  Weighted Score: {match['Weighted_Score']:.1f} | Semantic Score: {match['Semantic_Score']:.1f}\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {save_path}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("\n" + "="*80)
        print("PROPERTY MATCHING SYSTEM - FULL ANALYSIS")
        print("="*80)
        
        # Load and preprocess
        self.load_data()
        self.preprocess_data()
        
        # Calculate matches
        self.calculate_all_matches()
        
        # Generate visualizations
        self.create_match_heatmap()
        self.create_distribution_analysis()
        
        # Create charts for sample users
        for user_id in [1, 2, 3]:
            self.create_top_matches_chart(user_id)
        
        # Generate report
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  - match_heatmap.png")
        print("  - score_distribution.png")
        print("  - top_matches_user_1.png, top_matches_user_2.png, top_matches_user_3.png")
        print("  - match_summary_report.txt")
        print("\nRun 'streamlit run property_matcher.py' for interactive UI!")


# ============================================================================
# SECTION 4: STREAMLIT INTERACTIVE UI
# ============================================================================

def run_streamlit_ui():
    """Run the interactive Streamlit user interface."""
    
    st.set_page_config(
        page_title="Property Matching System",
        page_icon="🏠",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 20px;
        }
        .stMetric {
            background-color: #e6fffa;
            padding: 15px;
            border-radius: 10px;
        }
        .stMetric label, .stMetric div {
            color: #000000 !important;
        }
        h1 {
            color: #1f77b4;
            font-weight: bold;
        }
        h2, h3 {
            color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏠 Property Matching System with AI/ML")
    st.markdown("### Find the perfect property match based on user preferences")
    
    # Initialize system
    if 'matcher' not in st.session_state:
        with st.spinner("Loading data and initializing system..."):
            matcher = PropertyMatchingSystem()
            matcher.load_data()
            matcher.preprocess_data()
            matcher.calculate_all_matches()
            st.session_state.matcher = matcher
    
    matcher = st.session_state.matcher
    
    # Sidebar for user selection or custom input
    st.sidebar.header("🔍 Select Mode")
    mode = st.sidebar.radio("Choose matching mode:", ["Existing Users", "Custom Input"])
    
    if mode == "Existing Users":
        st.sidebar.header("👤 Select User")
        user_ids = sorted(matcher.user_df_processed['User ID'].unique())
        selected_user = st.sidebar.selectbox("User ID:", user_ids)
        
        # Get user information
        user_info = matcher.user_df_processed[matcher.user_df_processed['User ID'] == selected_user].iloc[0]
        
        # Display user preferences
        st.header(f"User {selected_user} Preferences")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Budget", user_info['Budget'])
        with col2:
            st.metric("Bedrooms", int(user_info['Bedrooms']))
        with col3:
            st.metric("Bathrooms", int(user_info['Bathrooms']))
        with col4:
            st.metric("User ID", selected_user)
        
        st.info(f"**Preferences:** {user_info['Qualitative Description']}")
        
        # Get top matches
        st.header("🎯 Top Property Matches")
        top_n = st.slider("Number of matches to display:", 5, 15, 10)
        top_matches = matcher.get_top_matches(selected_user, top_n=top_n)
        
        # Display matches
        for idx, match in top_matches.iterrows():
            with st.expander(f"🏡 Property {int(match['Property_ID'])} - Match Score: {match['Match_Score']:.1f}/100", expanded=(idx == top_matches.index[0])):
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Overall Score", f"{match['Match_Score']:.1f}")
                with col2:
                    st.metric("Price", match['Property_Price'])
                with col3:
                    st.metric("Bedrooms", int(match['Property_Bedrooms']))
                with col4:
                    st.metric("Bathrooms", int(match['Property_Bathrooms']))
                with col5:
                    st.metric("Area (sq ft)", int(match['Property_Area']))
                
                # Score breakdown
                st.markdown("**Score Breakdown:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.progress(match['Weighted_Score'] / 100)
                    st.caption(f"Numerical Match: {match['Weighted_Score']:.1f}/100")
                with col2:
                    st.progress(match['Semantic_Score'] / 100)
                    st.caption(f"Semantic Match: {match['Semantic_Score']:.1f}/100")
                
                # Property description
                prop_info = matcher.prop_df_processed[matcher.prop_df_processed['Property ID'] == match['Property_ID']].iloc[0]
                st.markdown(f"**Description:** {prop_info['Qualitative Description']}")
        
        # Visualization
        st.header("📊 Match Analysis")
        
        # Create a simple bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn(top_matches['Match_Score'].values / 100)
        ax.barh(top_matches['Property_ID'].astype(str), top_matches['Match_Score'], color=colors)
        ax.set_xlabel('Match Score', fontsize=12)
        ax.set_ylabel('Property ID', fontsize=12)
        ax.set_title(f'Property Match Scores for User {selected_user}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    else:  # Custom Input
        st.header("🔧 Custom Property Matching")
        st.markdown("Enter custom user preferences to find matching properties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_budget = st.number_input("Budget ($)", min_value=100000, max_value=2000000, value=500000, step=50000)
            custom_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
            custom_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
        
        with col2:
            custom_description = st.text_area(
                "Describe your ideal property:",
                value="I want a modern home with lots of natural light and a spacious kitchen.",
                height=150
            )
        
        if st.button("🔍 Find Matches", type="primary"):
            # Create a custom user row
            custom_user = pd.Series({
                'User ID': 999,
                'Budget': f"${custom_budget/1000:.0f}k",
                'Bedrooms': custom_bedrooms,
                'Bathrooms': custom_bathrooms,
                'Qualitative Description': custom_description,
                'Budget_Numeric': custom_budget,
                'Budget_Norm': (custom_budget - matcher.user_df_processed['Budget_Numeric'].min()) / 
                               (matcher.user_df_processed['Budget_Numeric'].max() - matcher.user_df_processed['Budget_Numeric'].min()),
                'Bedrooms_Norm': (custom_bedrooms - matcher.user_df_processed['Bedrooms'].min()) /
                                (matcher.user_df_processed['Bedrooms'].max() - matcher.user_df_processed['Bedrooms'].min()),
                'Bathrooms_Norm': (custom_bathrooms - matcher.user_df_processed['Bathrooms'].min()) /
                                 (matcher.user_df_processed['Bathrooms'].max() - matcher.user_df_processed['Bathrooms'].min())
            })
            
            # Calculate matches for custom user
            results = []
            for _, prop in matcher.prop_df_processed.iterrows():
                hybrid, weighted, semantic = matcher.calculate_hybrid_score(custom_user, prop)
                results.append({
                    'Property_ID': prop['Property ID'],
                    'Match_Score': hybrid,
                    'Weighted_Score': weighted,
                    'Semantic_Score': semantic,
                    'Property_Price': prop['Price'],
                    'Property_Bedrooms': prop['Bedrooms'],
                    'Property_Bathrooms': prop['Bathrooms'],
                    'Property_Area': prop['Living Area (sq ft)'],
                    'Property_Description': prop['Qualitative Description']
                })
            
            custom_matches = pd.DataFrame(results).nlargest(10, 'Match_Score')
            
            st.success(f"Found {len(custom_matches)} matching properties!")
            
            # Display custom matches
            for idx, match in custom_matches.iterrows():
                with st.expander(f"🏡 Property {int(match['Property_ID'])} - Match Score: {match['Match_Score']:.1f}/100"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall Score", f"{match['Match_Score']:.1f}")
                    with col2:
                        st.metric("Price", match['Property_Price'])
                    with col3:
                        st.metric("Bedrooms", int(match['Property_Bedrooms']))
                    with col4:
                        st.metric("Bathrooms", int(match['Property_Bathrooms']))
                    
                    st.markdown(f"**Description:** {match['Property_Description']}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Property Matching System** | Powered by AI/ML | Case Study 2026 | Ramakrishna_Product_Developer")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running with Streamlit
    try:
        # This will only work when run with Streamlit
        if STREAMLIT_AVAILABLE:
            run_streamlit_ui()
    except:
        # Not running with Streamlit, run batch analysis
        print("\nRunning batch analysis mode...")
        print("For interactive UI, run: streamlit run property_matcher.py\n")
        
        matcher = PropertyMatchingSystem()
        matcher.run_full_analysis()
