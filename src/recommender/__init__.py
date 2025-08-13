"""
Music recommendation module.

This module contains the core recommendation algorithms including content-based
filtering and playlist-based recommendations.
"""

# Import functions that don't require heavy dependencies
def recommend_with_normalized_features(*args, **kwargs):
    """Import and call the recommendation function."""
    from .content_based import recommend_with_normalized_features
    return recommend_with_normalized_features(*args, **kwargs)

__all__ = [
    "recommend_with_normalized_features",
]
