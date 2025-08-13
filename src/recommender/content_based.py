"""
Content-based recommendation module.

This module contains the core content-based recommendation algorithm that uses
normalized music features and cosine similarity to find similar songs.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Dict
import logging
from ..features.feature_utils import calculate_similarity

logger = logging.getLogger(__name__)


def recommend_with_normalized_features(query_idx: int, 
                                     normalized_matrix: np.ndarray, 
                                     track_names: List[str], 
                                     top_k: int = 4) -> List[Tuple[str, float]]:
    """
    Recommend similar songs using normalized features and cosine similarity.
    
    This is the core recommendation function that was the breakthrough in Week 1.
    It uses StandardScaler-normalized features to get realistic similarity scores.
    
    Args:
        query_idx (int): Index of the query song in the matrix
        normalized_matrix (np.ndarray): Normalized feature matrix
        track_names (List[str]): List of track names corresponding to matrix rows
        top_k (int): Number of recommendations to return
        
    Returns:
        List[Tuple[str, float]]: List of (track_name, similarity_score) tuples
        
    Raises:
        ValueError: If query_idx is out of bounds
        ValueError: If matrix dimensions don't match track_names
    """
    if query_idx >= len(track_names):
        raise ValueError(f"Query index {query_idx} out of bounds for {len(track_names)} tracks")
    
    if normalized_matrix.shape[0] != len(track_names):
        raise ValueError(f"Matrix has {normalized_matrix.shape[0]} rows but {len(track_names)} track names")
    
    try:
        # Calculate similarities
        query_embedding = normalized_matrix[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, normalized_matrix)[0]
        
        # Get indices sorted by similarity (descending)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Collect recommendations (excluding the query song itself)
        recommendations = []
        for idx in similar_indices:
            if idx != query_idx and len(recommendations) < top_k:
                track_name = track_names[idx]
                similarity_score = similarities[idx]
                recommendations.append((track_name, similarity_score))
        
        logger.info(f"Generated {len(recommendations)} recommendations for {track_names[query_idx]}")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise


def recommend_from_features(query_features: np.ndarray,
                           candidate_matrix: np.ndarray,
                           candidate_names: List[str],
                           top_k: int = 4) -> List[Tuple[str, float]]:
    """
    Recommend songs from a candidate set based on query features.
    
    This function is useful when you have features for a new song and want to
    find similar songs from a pre-computed candidate matrix.
    
    Args:
        query_features (np.ndarray): 29-dimensional feature vector for query song
        candidate_matrix (np.ndarray): Normalized feature matrix of candidate songs
        candidate_names (List[str]): Names of candidate songs
        top_k (int): Number of recommendations to return
        
    Returns:
        List[Tuple[str, float]]: List of (track_name, similarity_score) tuples
    """
    if query_features.shape[0] != 29:
        raise ValueError(f"Query features should be 29-dimensional, got {query_features.shape[0]}")
    
    if candidate_matrix.shape[0] != len(candidate_names):
        raise ValueError(f"Matrix has {candidate_matrix.shape[0]} rows but {len(candidate_names)} names")
    
    try:
        # Reshape query features for similarity calculation
        query_embedding = query_features.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, candidate_matrix)[0]
        
        # Get top-k recommendations
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in similar_indices:
            track_name = candidate_names[idx]
            similarity_score = similarities[idx]
            recommendations.append((track_name, similarity_score))
        
        logger.info(f"Generated {len(recommendations)} recommendations from {len(candidate_names)} candidates")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations from features: {str(e)}")
        raise


def print_recommendations(query_name: str, recommendations: List[Tuple[str, float]], 
                         title: Optional[str] = None, show_diversity: bool = False,
                         diversity_stats: Optional[Dict[str, float]] = None) -> None:
    """
    Print recommendations in a formatted way.
    
    Args:
        query_name (str): Name of the query song/playlist
        recommendations (List[Tuple[str, float]]): List of recommendations
        title (str, optional): Custom title for the output
        show_diversity (bool): Whether to show diversity statistics
        diversity_stats (Dict[str, float], optional): Playlist diversity statistics
    """
    if title:
        print(f"🎵 {title}: {query_name}")
    else:
        print(f"🎵 Songs similar to: {query_name}")
    
    print("=" * 60)
    
    if show_diversity and diversity_stats:
        print(f"📊 Playlist Diversity:")
        print(f"   - Diversity Score: {diversity_stats['diversity_score']:.3f}")
        print(f"   - Feature Variance: {diversity_stats['feature_variance']:.3f}")
        print(f"   - Track Count: {diversity_stats['track_count']}")
        print()
    
    print("🎯 Top Recommendations:")
    for i, (track_name, similarity_score) in enumerate(recommendations, 1):
        print(f"   {i:2d}. {track_name} (similarity: {similarity_score:.3f})")
    print()


def get_recommendation_quality_stats(recommendations: List[Tuple[str, float]]) -> dict:
    """
    Calculate quality statistics for a set of recommendations.
    
    Args:
        recommendations (List[Tuple[str, float]]): List of recommendations
        
    Returns:
        dict: Dictionary containing quality statistics
    """
    if not recommendations:
        return {}
    
    similarities = [score for _, score in recommendations]
    
    stats = {
        'count': len(recommendations),
        'mean_similarity': np.mean(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities),
        'std_similarity': np.std(similarities),
        'positive_count': sum(1 for s in similarities if s > 0),
        'negative_count': sum(1 for s in similarities if s < 0)
    }
    
    return stats



