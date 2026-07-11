"""
Streamlit Interactive UI
=========================
Interactive property matching: browse existing users' top matches, or enter
custom preferences for real-time matching.

Usage:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from matching_engine import PropertyMatchingSystem

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

    user_info = matcher.user_df_processed[matcher.user_df_processed['User ID'] == selected_user].iloc[0]

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

    st.header("🎯 Top Property Matches")
    top_n = st.slider("Number of matches to display:", 5, 15, 10)
    top_matches = matcher.get_top_matches(selected_user, top_n=top_n)

    for idx, match in top_matches.iterrows():
        with st.expander(f"🏡 Property {int(match['Property_ID'])} - Match Score: {match['Match_Score']:.1f}/100", expanded=(idx == top_matches.index[0])):
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

            st.markdown("**Score Breakdown:**")
            col1, col2 = st.columns(2)
            with col1:
                st.progress(match['Weighted_Score'] / 100)
                st.caption(f"Numerical Match: {match['Weighted_Score']:.1f}/100")
            with col2:
                st.progress(match['Semantic_Score'] / 100)
                st.caption(f"Semantic Match: {match['Semantic_Score']:.1f}/100")

            prop_info = matcher.prop_df_processed[matcher.prop_df_processed['Property ID'] == match['Property_ID']].iloc[0]
            st.markdown(f"**Description:** {prop_info['Qualitative Description']}")

    st.header("📊 Match Analysis")

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
