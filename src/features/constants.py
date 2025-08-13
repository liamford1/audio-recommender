"""
Constants for music feature extraction.

This module contains constants and simple functions that don't require
heavy dependencies like librosa.
"""

def get_feature_names() -> list:
    """
    Get the names of the 29 music features.
    
    Returns:
        list: List of feature names in the order they appear in the feature vector
    """
    return (['chroma_' + str(i) for i in range(12)] + 
            ['tempo'] + 
            ['spectral_centroid', 'spectral_rolloff', 'zero_crossing'] +
            ['mfcc_' + str(i) for i in range(13)])


# Feature breakdown
CHROMA_FEATURES = 12
TEMPO_FEATURES = 1
SPECTRAL_FEATURES = 3
MFCC_FEATURES = 13
TOTAL_FEATURES = CHROMA_FEATURES + TEMPO_FEATURES + SPECTRAL_FEATURES + MFCC_FEATURES

# Feature groups
FEATURE_GROUPS = {
    'chroma': list(range(CHROMA_FEATURES)),
    'tempo': [CHROMA_FEATURES],
    'spectral': list(range(CHROMA_FEATURES + 1, CHROMA_FEATURES + 1 + SPECTRAL_FEATURES)),
    'mfcc': list(range(CHROMA_FEATURES + 1 + SPECTRAL_FEATURES, TOTAL_FEATURES))
}
