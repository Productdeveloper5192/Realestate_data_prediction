"""
Visualization & Reporting
==========================
Chart and text-report generation for property match results. Each function
takes a `PropertyMatchingSystem` that has already run `calculate_all_matches()`.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_match_heatmap(matcher, save_path='match_heatmap.png'):
    """Create a heatmap showing match scores for all user-property pairs."""
    print("\nGenerating match score heatmap...")

    heatmap_data = matcher.match_scores.pivot(
        index='User_ID',
        columns='Property_ID',
        values='Match_Score'
    )

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


def create_top_matches_chart(matcher, user_id, save_path=None):
    """Create a bar chart showing top property matches for a user."""
    top_matches = matcher.get_top_matches(user_id, top_n=10)

    if save_path is None:
        save_path = f'top_matches_user_{user_id}.png'

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

    ax2.bar(x - width / 2, top_matches['Weighted_Score'], width, label='Weighted (Numerical)', alpha=0.8)
    ax2.bar(x + width / 2, top_matches['Semantic_Score'], width, label='Semantic (Text)', alpha=0.8)

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


def create_distribution_analysis(matcher, save_path='score_distribution.png'):
    """Analyze and visualize the distribution of match scores."""
    print("\nGenerating score distribution analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall match score distribution
    axes[0, 0].hist(matcher.match_scores['Match_Score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Match Score', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of All Match Scores', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(matcher.match_scores['Match_Score'].mean(), color='red', linestyle='--', label=f'Mean: {matcher.match_scores["Match_Score"].mean():.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Average match score by user
    avg_by_user = matcher.match_scores.groupby('User_ID')['Match_Score'].mean().sort_values(ascending=False)
    axes[0, 1].bar(avg_by_user.index.astype(str), avg_by_user.values, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('User ID', fontsize=11)
    axes[0, 1].set_ylabel('Average Match Score', fontsize=11)
    axes[0, 1].set_title('Average Match Score by User', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Average match score by property
    avg_by_prop = matcher.match_scores.groupby('Property_ID')['Match_Score'].mean().sort_values(ascending=False).head(15)
    axes[1, 0].bar(avg_by_prop.index.astype(str), avg_by_prop.values, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Property ID', fontsize=11)
    axes[1, 0].set_ylabel('Average Match Score', fontsize=11)
    axes[1, 0].set_title('Top 15 Properties by Average Match Score', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Weighted vs Semantic correlation
    axes[1, 1].scatter(matcher.match_scores['Weighted_Score'], matcher.match_scores['Semantic_Score'],
                        alpha=0.5, c=matcher.match_scores['Match_Score'], cmap='viridis')
    axes[1, 1].set_xlabel('Weighted Score (Numerical)', fontsize=11)
    axes[1, 1].set_ylabel('Semantic Score (Text)', fontsize=11)
    axes[1, 1].set_title('Correlation: Numerical vs Semantic Matching', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    corr = matcher.match_scores['Weighted_Score'].corr(matcher.match_scores['Semantic_Score'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                     transform=axes[1, 1].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distribution analysis saved to {save_path}")
    plt.close()


def generate_summary_report(matcher, save_path='match_summary_report.txt'):
    """Generate a comprehensive text summary of the matching results."""
    print("\nGenerating summary report...")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PROPERTY MATCHING SYSTEM - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Users: {len(matcher.user_df_processed)}\n")
        f.write(f"Total Properties: {len(matcher.prop_df_processed)}\n")
        f.write(f"Total Match Scores Calculated: {len(matcher.match_scores)}\n\n")

        f.write(f"Average Match Score: {matcher.match_scores['Match_Score'].mean():.2f}\n")
        f.write(f"Median Match Score: {matcher.match_scores['Match_Score'].median():.2f}\n")
        f.write(f"Std Dev: {matcher.match_scores['Match_Score'].std():.2f}\n")
        f.write(f"Min Score: {matcher.match_scores['Match_Score'].min():.2f}\n")
        f.write(f"Max Score: {matcher.match_scores['Match_Score'].max():.2f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 3 PROPERTY MATCHES FOR EACH USER\n")
        f.write("=" * 80 + "\n\n")

        for user_id in sorted(matcher.user_df_processed['User ID'].unique()):
            user_info = matcher.user_df_processed[matcher.user_df_processed['User ID'] == user_id].iloc[0]
            top_matches = matcher.get_top_matches(user_id, top_n=3)

            f.write(f"\nUSER {user_id}\n")
            f.write(f"Budget: {user_info['Budget']}, Bedrooms: {user_info['Bedrooms']}, Bathrooms: {user_info['Bathrooms']}\n")
            f.write(f"Preferences: {user_info['Qualitative Description'][:150]}...\n")
            f.write("-" * 80 + "\n")

            for idx, match in top_matches.iterrows():
                f.write(f"\n  Property {match['Property_ID']} - Match Score: {match['Match_Score']:.1f}\n")
                f.write(f"  Price: {match['Property_Price']}, Bedrooms: {int(match['Property_Bedrooms'])}, ")
                f.write(f"Bathrooms: {int(match['Property_Bathrooms'])}, Area: {int(match['Property_Area'])} sq ft\n")
                f.write(f"  Weighted Score: {match['Weighted_Score']:.1f} | Semantic Score: {match['Semantic_Score']:.1f}\n")

            f.write("\n")

    print(f"Summary report saved to {save_path}")
