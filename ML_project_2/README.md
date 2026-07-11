# Property Matching System with AI/ML

## 🎯 Project Overview

A comprehensive property matching system that uses AI/ML techniques to match properties with user preferences and calculate match scores. This solution processes user preferences and property characteristics to provide intelligent recommendations.

## ✨ Features

- **Advanced Matching Algorithms**:
  - Weighted numerical scoring (budget, bedrooms, bathrooms, area)
  - Semantic text matching using TF-IDF and cosine similarity
  - Hybrid approach combining both methods

- **Comprehensive Visualizations**:
  - Match score heatmap for all user-property pairs
  - Distribution analysis and statistics
  - Top property matches for individual users
  - Score breakdown (numerical vs semantic)

- **Interactive UI**:
  - Real-time property matching with Streamlit
  - Support for existing users and custom input
  - Detailed match explanations

## 📊 Methodology

### Match Score Calculation

The system uses a **hybrid approach** combining:

1. **Weighted Numerical Matching (70% weight)**:
   - Budget/Price: 40% (penalties for over-budget properties)
   - Bedrooms: 25% (exact match preferred)
   - Bathrooms: 20% (exact match preferred)
   - Living Area: 15% (bonus for larger spaces)

2. **Semantic Text Matching (30% weight)**:
   - TF-IDF vectorization of qualitative descriptions
   - Cosine similarity between user preferences and property descriptions
   - Captures contextual and qualitative aspects

### Final Score Formula
```
Match Score = (Weighted Numerical Score × 0.70) + (Semantic Similarity × 0.30)
Range: 0-100
```

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Batch Analysis
Generate all visualizations and reports:
```bash
python property_matcher.py
```

This creates:
- `match_heatmap.png` - Heatmap of all user-property match scores
- `score_distribution.png` - Statistical analysis of match scores
- `top_matches_user_X.png` - Top matches for individual users
- `match_summary_report.txt` - Detailed text report

### Run Interactive UI
Launch the Streamlit interface:
```bash
streamlit run property_matcher.py
```

Features:
- Browse existing users and see their top property matches
- Enter custom preferences for real-time matching
- View detailed score breakdowns
- Interactive visualizations

## 📁 File Structure

```
ML_project_2/
├── property_matcher.py          # Single comprehensive solution file
├── Case Study 2 Data.xlsx       # Input data (User & Property)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── Generated outputs:
    ├── match_heatmap.png
    ├── score_distribution.png
    ├── top_matches_user_*.png
    └── match_summary_report.txt
```

## 📈 Key Results

- **Total Match Scores**: 750 (25 users × 30 properties)
- **Average Match Score**: 38.88/100
- **Score Range**: 6.95 - 70.43
- **Correlation**: Numerical and semantic scores show 0.095 correlation

### Best Performing Properties
Properties with highest average match scores across all users:
1. Property 18 - 2BR/1BA at $300k
2. Property 30 - 4BR/3BA at $500k  
3. Property 9 - 4BR/3BA at $400k

## 🧮 Technical Implementation

### Data Processing
- Price parsing from string format ($500k → 500000)
- Feature normalization using MinMaxScaler
- Missing value handling (none found in current dataset)

### Machine Learning Techniques
- **Scikit-learn**: Preprocessing, TF-IDF, cosine similarity
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Advanced visualizations

### Algorithm Complexity
- Time Complexity: O(U × P) where U = users, P = properties
- Space Complexity: O(U × P) for storing match scores
- Current: 750 comparisons in < 5 seconds

## 💡 Design Decisions

1. **Hybrid Approach**: Combines numerical precision with semantic understanding
2. **Budget Penalty**: Properties over budget receive heavy penalties
3. **Text Matching**: Captures qualitative preferences like "modern", "cozy", "spacious"
4. **Weighted Scoring**: Price matters most (40%), followed by bedrooms/bathrooms
5. **Single File Architecture**: All code in one file for easy deployment and sharing

## 🎨 Visualizations

### 1. Match Score Heatmap
Shows all 750 user-property combinations with color-coded scores (green = high match, red = low match)

### 2. Distribution Analysis
Four-panel visualization:
- Overall score distribution histogram
- Average scores by user
- Top properties by average score
- Correlation between numerical and semantic matching

### 3. Individual User Charts
Horizontal bar charts showing:
- Top 10 property matches per user
- Score breakdown (weighted vs semantic)

## 🔧 Customization

The system is highly customizable through the `PropertyMatchingSystem` class:

```python
# Adjust weights in calculate_weighted_score()
price_score * 0.40    # Budget importance
bedroom_score * 0.25  # Bedroom importance
bathroom_score * 0.20 # Bathroom importance
area_score * 0.15     # Living area importance

# Adjust hybrid ratio in calculate_hybrid_score()
weighted * 0.70 + semantic * 0.30
```

## 📝 Presentation Outline

1. **Introduction**: Problem statement and objectives
2. **Data Overview**: User preferences and property characteristics
3. **Methodology**: Hybrid matching approach
4. **Results**: Heatmaps, distributions, top matches
5. **Live Demo**: Interactive UI demonstration
6. **Conclusion**: Findings and future improvements

## 🚧 Future Enhancements

- Add location-based matching (geospatial analysis)
- Integrate LLM (GPT/Gemini) for advanced semantic understanding
- Add amenity matching (pool, gym, parking)
- Implement collaborative filtering
- Add property image analysis with computer vision
- Support multi-criteria decision analysis (MCDA)

## 📄 License

This project is developed for educational purposes as part of a data science case study.

## 👤 Author

Agent Mira - Data Science Case Study 2026
**Product Developer:** Ramakrishna_Product_Developer

---

**Note**: All code is contained in a **single file** (`property_matcher.py`) for easy sharing and deployment.
