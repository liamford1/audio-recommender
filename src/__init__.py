"""
Music Recommender Package

A music recommendation system that analyzes Spotify playlists and recommends 
similar songs based on audio characteristics.
"""

__version__ = "0.1.0"
__author__ = "Liam Ford"

# Import functions that don't require heavy dependencies
def get_feature_names():
    """Get the names of the 29 music features."""
    from .features.constants import get_feature_names
    return get_feature_names()

__all__ = [
    "get_feature_names",
]
