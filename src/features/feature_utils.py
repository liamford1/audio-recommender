"""
Feature utilities module.

This module contains utility functions for feature normalization, similarity calculation,
and feature matrix operations.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def normalize_features(feature_matrix: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    This is the critical breakthrough that fixed the similarity calculation.
    Raw features gave false similarities due to scale differences.
    
    Args:
        feature_matrix (np.ndarray): Matrix of features (n_samples, n_features)
        scaler (StandardScaler, optional): Pre-fitted scaler. If None, fit new scaler
        
    Returns:
        Tuple[np.ndarray, StandardScaler]: Normalized features and fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(feature_matrix)
        logger.info("Fitted new StandardScaler to feature matrix")
    else:
        normalized_matrix = scaler.transform(feature_matrix)
        logger.info("Applied pre-fitted StandardScaler to feature matrix")
    
    return normalized_matrix, scaler


def normalize_features_simple(feature_list: List[np.ndarray]) -> np.ndarray:
    """
    Simple normalization function that takes a list of feature vectors.
    
    This is a convenience function for the test that matches the expected interface.
    
    Args:
        feature_list (List[np.ndarray]): List of feature vectors
        
    Returns:
        np.ndarray: Normalized feature matrix
    """
    feature_matrix = np.array(feature_list)
    normalized_matrix, _ = normalize_features(feature_matrix)
    return normalized_matrix


def calculate_similarity(features1: Union[np.ndarray, int], 
                        features2: Optional[np.ndarray] = None,
                        feature_matrix: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
    """
    Unified similarity calculation function.
    
    This function can handle two different use cases:
    1. Calculate similarity between two feature vectors (returns float)
    2. Calculate similarity between a query song and all songs in a matrix (returns array)
    
    Args:
        features1 (np.ndarray or int): Either a feature vector or query index
        features2 (np.ndarray, optional): Second feature vector (for case 1)
        feature_matrix (np.ndarray, optional): Feature matrix (for case 2)
        
    Returns:
        float or np.ndarray: Similarity score(s)
    """
    if features2 is not None and feature_matrix is not None:
        raise ValueError("Cannot specify both features2 and feature_matrix")
    
    if features2 is not None:
        # Case 1: Two feature vectors
        if features1.shape != features2.shape:
            raise ValueError(f"Feature vectors must have same shape, got {features1.shape} and {features2.shape}")
        
        if len(features1.shape) == 1:
            features1 = features1.reshape(1, -1)
            features2 = features2.reshape(1, -1)
        
        similarity = cosine_similarity(features1, features2)[0, 0]
        logger.info(f"Calculated similarity: {similarity:.3f}")
        return similarity
    
    elif feature_matrix is not None:
        # Case 2: Query index and feature matrix
        if isinstance(features1, int):
            query_idx = features1
            query_embedding = feature_matrix[query_idx].reshape(1, -1)
            similarities = cosine_similarity(query_embedding, feature_matrix)[0]
            return similarities
        else:
            raise ValueError("When using feature_matrix, features1 must be an integer index")
    
    else:
        raise ValueError("Must specify either features2 or feature_matrix")


def get_similarity_statistics(similarity_matrix: np.ndarray) -> dict:
    """
    Calculate statistics from a similarity matrix.
    
    Args:
        similarity_matrix (np.ndarray): Matrix of similarity scores
        
    Returns:
        dict: Dictionary containing similarity statistics
    """
    # Get upper triangle (excluding diagonal)
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    stats = {
        'mean': np.mean(upper_triangle),
        'std': np.std(upper_triangle),
        'min': np.min(upper_triangle),
        'max': np.max(upper_triangle),
        'median': np.median(upper_triangle)
    }
    
    return stats


def save_feature_components(feature_matrix: np.ndarray, 
                          normalized_matrix: np.ndarray,
                          scaler: StandardScaler,
                          track_info: List[str],
                          output_dir: str) -> None:
    """
    Save all feature components for future use.
    
    Args:
        feature_matrix (np.ndarray): Raw feature matrix
        normalized_matrix (np.ndarray): Normalized feature matrix
        scaler (StandardScaler): Fitted scaler
        track_info (List[str]): List of track names/info
        output_dir (str): Directory to save components
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save matrices
    np.save(f"{output_dir}/music_features_raw.npy", feature_matrix)
    np.save(f"{output_dir}/music_features_normalized.npy", normalized_matrix)
    
    # Save scaler
    with open(f"{output_dir}/feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save track info
    with open(f"{output_dir}/music_track_info.pkl", 'wb') as f:
        pickle.dump(track_info, f)
    
    logger.info(f"Saved feature components to {output_dir}")


def load_feature_components(input_dir: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Load previously saved feature components.
    
    Args:
        input_dir (str): Directory containing saved components
        
    Returns:
        Tuple: (raw_matrix, normalized_matrix, scaler, track_info)
    """
    # Load matrices
    raw_matrix = np.load(f"{input_dir}/music_features_raw.npy")
    normalized_matrix = np.load(f"{input_dir}/music_features_normalized.npy")
    
    # Load scaler
    with open(f"{input_dir}/feature_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Load track info
    with open(f"{input_dir}/music_track_info.pkl", 'rb') as f:
        track_info = pickle.load(f)
    
    logger.info(f"Loaded feature components from {input_dir}")
    return raw_matrix, normalized_matrix, scaler, track_info


def validate_feature_matrix(feature_matrix: np.ndarray) -> bool:
    """
    Validate that a feature matrix is properly formatted.
    
    Args:
        feature_matrix (np.ndarray): Feature matrix to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if feature_matrix.ndim != 2:
        logger.error(f"Feature matrix should be 2D, got {feature_matrix.ndim}D")
        return False
    
    if feature_matrix.shape[1] != 29:
        logger.error(f"Feature matrix should have 29 features, got {feature_matrix.shape[1]}")
        return False
    
    if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
        logger.error("Feature matrix contains NaN or infinite values")
        return False
    
    return True
