"""
Audio feature extraction module.

This module contains functions for extracting music-specific features from audio files,
including chroma, tempo, spectral, and MFCC features.
"""

# Import functions that don't require heavy dependencies
def get_feature_names():
    """Get the names of the 29 music features."""
    from .constants import get_feature_names
    return get_feature_names()

__all__ = [
    "get_feature_names",
]
