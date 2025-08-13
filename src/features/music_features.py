"""
Music feature extraction module.

This module contains the core function for extracting 29-dimensional music features
from audio files, including chroma, tempo, spectral, and MFCC features.
"""

import os
import numpy as np
from typing import Union, Optional
import logging
from .constants import get_feature_names

# Import librosa only when needed
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_music_features(file_path: str, duration: int = 30) -> np.ndarray:
    """
    Extract music-specific features from an audio file.
    
    This function extracts 29-dimensional music features including:
    - 12 chroma features (harmonic content)
    - 1 tempo feature (rhythm)
    - 3 spectral features (brightness, energy distribution)
    - 13 MFCC features (timbre/texture)
    
    Args:
        file_path (str): Path to the audio file
        duration (int): Duration in seconds to analyze (default: 30)
        
    Returns:
        np.ndarray: 29-dimensional feature vector
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the audio file cannot be loaded
        ImportError: If librosa is not available
        Exception: For other processing errors
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for feature extraction. Install with: pip install librosa")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        logger.info(f"Processing: {os.path.basename(file_path)}")
        
        # Load audio file
        y, sr = librosa.load(file_path, duration=duration)
        
        # HARMONIC CONTENT (what chords/keys)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # 12 values
        
        # RHYTHM/TEMPO (how it feels rhythmically)  
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_features = [tempo.item() if hasattr(tempo, 'item') else float(tempo)]
        
        # SPECTRAL SHAPE (brightness, energy distribution)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # TIMBRAL TEXTURE (what instruments/voices sound like)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # 13 values
        
        # COMBINE ALL FEATURES
        music_vector = np.concatenate([
            chroma_mean,           # 12 harmony features
            tempo_features,        # 1 tempo feature  
            [spectral_centroid, spectral_rolloff, zero_crossing],  # 3 spectral
            mfcc_mean             # 13 timbre features
        ])
        
        # Validate output
        if len(music_vector) != 29:
            raise ValueError(f"Expected 29 features, got {len(music_vector)}")
        
        if np.any(np.isnan(music_vector)) or np.any(np.isinf(music_vector)):
            raise ValueError("Feature vector contains NaN or infinite values")
        
        return music_vector
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        raise
