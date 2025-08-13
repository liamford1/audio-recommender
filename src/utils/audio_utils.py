"""
Audio utility functions.

This module contains utility functions for audio file processing,
format validation, and other audio-related operations.
"""

import os
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def find_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find all audio files in a directory recursively.
    
    Args:
        directory (str): Directory to search
        extensions (List[str], optional): List of file extensions to include
        
    Returns:
        List[str]: List of paths to audio files
    """
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(audio_files)} audio files in {directory}")
    return audio_files


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that a file is a valid audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        bool: True if valid audio file, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file size (should be > 1KB)
    if os.path.getsize(file_path) < 1024:
        return False
    
    # Check file extension
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    if not any(file_path.lower().endswith(ext) for ext in audio_extensions):
        return False
    
    return True


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        float: Duration in seconds, or None if error
    """
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        return duration
    except Exception as e:
        logger.warning(f"Could not get duration for {file_path}: {e}")
        return None


def resample_audio(audio_data: np.ndarray, 
                  original_sr: int, 
                  target_sr: int) -> np.ndarray:
    """
    Resample audio data to a target sample rate.
    
    Args:
        audio_data (np.ndarray): Audio data
        original_sr (int): Original sample rate
        target_sr (int): Target sample rate
        
    Returns:
        np.ndarray: Resampled audio data
    """
    if original_sr == target_sr:
        return audio_data
    
    try:
        import librosa
        resampled = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
        return resampled
    except Exception as e:
        logger.error(f"Error resampling audio: {e}")
        raise


def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to a target dB level.
    
    Args:
        audio_data (np.ndarray): Audio data
        target_db (float): Target dB level
        
    Returns:
        np.ndarray: Normalized audio data
    """
    try:
        import librosa
        normalized = librosa.util.normalize(audio_data)
        
        # Convert to dB and adjust to target level
        current_db = 20 * np.log10(np.max(np.abs(normalized)) + 1e-10)
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        return normalized * gain_linear
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        raise


def extract_audio_segment(audio_data: np.ndarray, 
                         sr: int, 
                         start_time: float, 
                         duration: float) -> np.ndarray:
    """
    Extract a segment from audio data.
    
    Args:
        audio_data (np.ndarray): Audio data
        sr (int): Sample rate
        start_time (float): Start time in seconds
        duration (float): Duration in seconds
        
    Returns:
        np.ndarray: Audio segment
    """
    start_sample = int(start_time * sr)
    end_sample = int((start_time + duration) * sr)
    
    # Ensure bounds are valid
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)
    
    if start_sample >= end_sample:
        raise ValueError("Invalid time range for audio segment")
    
    return audio_data[start_sample:end_sample]


def save_audio_segment(audio_data: np.ndarray, 
                      sr: int, 
                      output_path: str,
                      format: str = 'wav') -> bool:
    """
    Save audio data to a file.
    
    Args:
        audio_data (np.ndarray): Audio data
        sr (int): Sample rate
        output_path (str): Output file path
        format (str): Output format ('wav', 'mp3', etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import soundfile as sf
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save audio file
        sf.write(output_path, audio_data, sr)
        
        logger.info(f"Saved audio to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving audio to {output_path}: {e}")
        return False


def get_audio_statistics(audio_data: np.ndarray) -> dict:
    """
    Calculate basic statistics for audio data.
    
    Args:
        audio_data (np.ndarray): Audio data
        
    Returns:
        dict: Dictionary of audio statistics
    """
    stats = {
        'length_samples': len(audio_data),
        'duration_seconds': len(audio_data) / 22050,  # Assuming 22.05kHz
        'mean': np.mean(audio_data),
        'std': np.std(audio_data),
        'min': np.min(audio_data),
        'max': np.max(audio_data),
        'rms': np.sqrt(np.mean(audio_data**2)),
        'zero_crossings': np.sum(np.diff(np.sign(audio_data)) != 0)
    }
    
    return stats


def detect_silence(audio_data: np.ndarray, 
                  threshold: float = 0.01,
                  min_duration: float = 0.1) -> List[Tuple[float, float]]:
    """
    Detect silent segments in audio data.
    
    Args:
        audio_data (np.ndarray): Audio data
        threshold (float): Amplitude threshold for silence
        min_duration (float): Minimum duration for silence detection (seconds)
        
    Returns:
        List[Tuple[float, float]]: List of (start_time, end_time) tuples
    """
    sr = 22050  # Assuming standard sample rate
    
    # Find samples below threshold
    silent_samples = np.abs(audio_data) < threshold
    
    # Find silent segments
    silent_segments = []
    start_sample = None
    
    for i, is_silent in enumerate(silent_samples):
        if is_silent and start_sample is None:
            start_sample = i
        elif not is_silent and start_sample is not None:
            end_sample = i
            duration = (end_sample - start_sample) / sr
            
            if duration >= min_duration:
                start_time = start_sample / sr
                end_time = end_sample / sr
                silent_segments.append((start_time, end_time))
            
            start_sample = None
    
    # Handle case where audio ends with silence
    if start_sample is not None:
        duration = (len(audio_data) - start_sample) / sr
        if duration >= min_duration:
            start_time = start_sample / sr
            end_time = len(audio_data) / sr
            silent_segments.append((start_time, end_time))
    
    return silent_segments
