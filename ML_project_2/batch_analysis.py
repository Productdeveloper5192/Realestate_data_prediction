"""
Batch Analysis Entry Point
============================
Runs the full matching pipeline and writes every chart/report to disk.

Usage:
    python batch_analysis.py
"""

from matching_engine import PropertyMatchingSystem
from visualization import (
    create_distribution_analysis,
    create_match_heatmap,
    create_top_matches_chart,
    generate_summary_report,
)


def run_full_analysis():
    """Run the complete analysis pipeline."""
    print("\n" + "=" * 80)
    print("PROPERTY MATCHING SYSTEM - FULL ANALYSIS")
    print("=" * 80)

    matcher = PropertyMatchingSystem()
    matcher.load_data()
    matcher.preprocess_data()
    matcher.calculate_all_matches()

    create_match_heatmap(matcher)
    create_distribution_analysis(matcher)

    for user_id in [1, 2, 3]:
        create_top_matches_chart(matcher, user_id)

    generate_summary_report(matcher)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - match_heatmap.png")
    print("  - score_distribution.png")
    print("  - top_matches_user_1.png, top_matches_user_2.png, top_matches_user_3.png")
    print("  - match_summary_report.txt")
    print("\nRun 'streamlit run app.py' for the interactive UI!")


if __name__ == "__main__":
    run_full_analysis()
