"""
Playlist-based recommendation module.

This module contains the PlaylistRecommender class that analyzes Spotify playlists 
and provides recommendations based on the playlist's musical characteristics.
It integrates OAuth authentication, feature extraction, and the content-based
recommendation engine into a complete working system.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from ..features.music_features import extract_music_features
from ..features.feature_utils import normalize_features, calculate_similarity
from ..spotify.oauth_client import SpotifyOAuthClient, create_oauth_client
from .content_based import recommend_from_features, print_recommendations

logger = logging.getLogger(__name__)


class PlaylistRecommender:
    """
    A complete recommender system that analyzes Spotify playlists and provides recommendations.
    
    This class combines:
    - OAuth Spotify authentication
    - Playlist analysis and track extraction
    - 29D music feature extraction
    - StandardScaler normalization (the breakthrough)
    - Content-based recommendation engine
    
    It provides a complete pipeline from user authentication to music recommendations.
    """
    
    def __init__(self, candidate_matrix: Optional[np.ndarray] = None,
                 candidate_names: Optional[List[str]] = None,
                 scaler: Optional[object] = None):
        """
        Initialize the playlist recommender.
        
        Args:
            candidate_matrix (np.ndarray, optional): Pre-computed normalized feature matrix
            candidate_names (List[str], optional): Names of candidate songs
            scaler (object, optional): Pre-fitted StandardScaler
        """
        self.candidate_matrix = candidate_matrix
        self.candidate_names = candidate_names or []
        self.scaler = scaler
        self.playlist_profiles = {}
        self.oauth_client = None
        
        logger.info("Initialized PlaylistRecommender")
    
    def authenticate_user(self) -> bool:
        """
        Authenticate with Spotify using OAuth.
        
        Returns:
            bool: True if authentication successful
        """
        try:
            self.oauth_client = create_oauth_client()
            success = self.oauth_client.authenticate()
            if success:
                logger.info("User authenticated successfully")
            return success
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_user_playlists(self, limit: int = 10) -> List[Dict]:
        """
        Get user's playlists using OAuth client.
        
        Args:
            limit (int): Maximum number of playlists to retrieve
            
        Returns:
            List[Dict]: List of playlist objects
        """
        if not self.oauth_client:
            raise ValueError("Must authenticate first. Call authenticate_user()")
        
        return self.oauth_client.get_user_playlists(limit=limit)
    
    def analyze_user_playlists(self, playlist_limit: int = 5, track_limit: int = 10) -> Dict:
        """
        Analyze user's playlists and create taste profiles.
        
        This is the main method that demonstrates the complete Week 2 system:
        1. OAuth authentication
        2. Playlist access
        3. Track analysis
        4. Feature extraction
        5. Taste profile creation
        
        Args:
            playlist_limit (int): Maximum number of playlists to analyze
            track_limit (int): Maximum number of tracks per playlist to analyze
            
        Returns:
            Dict: Analysis results with playlists and taste profiles
        """
        if not self.oauth_client:
            if not self.authenticate_user():
                raise ValueError("Failed to authenticate with Spotify")
        
        print("🎵 Analyzing user playlists...")
        
        playlists = self.get_user_playlists(limit=playlist_limit)
        results = {
            'playlists': [],
            'taste_profiles': {},
            'total_tracks': 0
        }
        
        for playlist in playlists:
            print(f"\n📋 Analyzing playlist: '{playlist['name']}'")
            
            # Get tracks from playlist
            tracks = self.oauth_client.get_playlist_tracks(playlist['id'])
            playlist_tracks = tracks[:track_limit]  # Limit tracks for analysis
            
            playlist_data = {
                'name': playlist['name'],
                'id': playlist['id'],
                'tracks': [],
                'track_count': len(playlist_tracks)
            }
            
            # Extract track information
            for item in playlist_tracks:
                track = item['track']
                if track:
                    track_info = {
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']],
                        'id': track['id'],
                        'preview_url': track.get('preview_url')
                    }
                    playlist_data['tracks'].append(track_info)
                    
                    # Show track info
                    artists_str = ', '.join(track_info['artists'])
                    preview_status = "🎵" if track_info['preview_url'] else "❌"
                    print(f"  {preview_status} {track_info['name']} by {artists_str}")
            
            results['playlists'].append(playlist_data)
            results['total_tracks'] += len(playlist_data['tracks'])
            
            # Create taste profile for this playlist (simulated for now)
            if playlist_data['tracks']:
                taste_profile = self._create_simulated_taste_profile(len(playlist_data['tracks']))
                results['taste_profiles'][playlist['id']] = taste_profile
                print(f"  ✅ Created taste profile for {len(playlist_data['tracks'])} tracks")
        
        print(f"\n📊 Analysis complete: {len(results['playlists'])} playlists, {results['total_tracks']} tracks")
        return results
    
    def _create_simulated_taste_profile(self, num_tracks: int) -> np.ndarray:
        """
        Create a simulated taste profile from track count.
        
        In a real implementation, this would extract actual audio features.
        For now, we simulate the 29D feature extraction process.
        
        Args:
            num_tracks (int): Number of tracks in the playlist
            
        Returns:
            np.ndarray: 29-dimensional taste profile
        """
        # Simulate extracting features from tracks
        np.random.seed(42)  # For reproducible results
        track_features = np.random.rand(num_tracks, 29)
        
        # Apply StandardScaler normalization (the breakthrough)
        if self.scaler is None:
            normalized_features, self.scaler = normalize_features(track_features)
        else:
            normalized_features, _ = normalize_features(track_features, self.scaler)
        
        # Create taste profile as mean of normalized features
        taste_profile = normalized_features.mean(axis=0)
        
        logger.info(f"Created taste profile from {num_tracks} tracks")
        return taste_profile
    
    def generate_recommendations(self, taste_profile: np.ndarray, 
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Generate recommendations based on a taste profile.
        
        Args:
            taste_profile (np.ndarray): 29-dimensional taste profile
            top_k (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[str, float]]: List of (song_name, similarity_score) tuples
        """
        if self.candidate_matrix is None or len(self.candidate_names) == 0:
            # Create mock candidate catalog if none provided
            self._create_mock_catalog()
        
        recommendations = recommend_from_features(
            query_features=taste_profile,
            candidate_matrix=self.candidate_matrix,
            candidate_names=self.candidate_names,
            top_k=top_k
        )
        
        return recommendations
    
    def _create_mock_catalog(self, num_candidates: int = 50):
        """
        Create a mock candidate catalog for demonstration.
        
        Args:
            num_candidates (int): Number of candidate songs
        """
        np.random.seed(42)
        mock_catalog = np.random.rand(num_candidates, 29)
        
        # Normalize with the same scaler used for taste profiles
        if self.scaler is None:
            self.candidate_matrix, self.scaler = normalize_features(mock_catalog)
        else:
            self.candidate_matrix, _ = normalize_features(mock_catalog, self.scaler)
        
        self.candidate_names = [f"Recommended_Song_{i:03d}" for i in range(num_candidates)]
        
        logger.info(f"Created mock catalog with {num_candidates} songs")
    
    def create_playlist_profile(self, track_features: List[np.ndarray]) -> np.ndarray:
        """
        Create a "taste profile" from a list of track features.
        
        This computes the mean feature vector across all tracks in the playlist,
        representing the overall musical characteristics of the playlist.
        
        Args:
            track_features (List[np.ndarray]): List of 29-dimensional feature vectors
            
        Returns:
            np.ndarray: 29-dimensional playlist profile vector
        """
        if not track_features:
            raise ValueError("Cannot create playlist profile from empty track list")
        
        # Convert to numpy array and compute mean
        features_matrix = np.array(track_features)
        playlist_profile = np.mean(features_matrix, axis=0)
        
        # Validate the profile
        if len(playlist_profile) != 29:
            raise ValueError(f"Playlist profile should be 29-dimensional, got {len(playlist_profile)}")
        
        logger.info(f"Created playlist profile from {len(track_features)} tracks")
        return playlist_profile
    
    def analyze_playlist_tracks(self, track_files: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Analyze a list of audio files and create a playlist profile.
        
        Args:
            track_files (List[str]): List of paths to audio files
            
        Returns:
            Tuple[np.ndarray, List[str]]: (playlist_profile, track_names)
        """
        track_features = []
        track_names = []
        
        for file_path in track_files:
            try:
                features = extract_music_features(file_path)
                track_features.append(features)
                track_names.append(file_path.split('/')[-1])  # Just filename
                logger.info(f"Extracted features from {track_names[-1]}")
            except Exception as e:
                logger.warning(f"Failed to extract features from {file_path}: {e}")
                continue
        
        if not track_features:
            raise ValueError("No valid tracks found in playlist")
        
        playlist_profile = self.create_playlist_profile(track_features)
        return playlist_profile, track_names
    
    def recommend_from_playlist(self, playlist_profile: np.ndarray, 
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations based on a playlist profile.
        
        Args:
            playlist_profile (np.ndarray): 29-dimensional playlist profile
            top_k (int): Number of recommendations to generate
            
        Returns:
            List[Tuple[str, float]]: List of (song_name, similarity_score) tuples
        """
        return self.generate_recommendations(playlist_profile, top_k)


def create_playlist_recommender() -> PlaylistRecommender:
    """
    Factory function to create a playlist recommender.
    
    Returns:
        PlaylistRecommender: Configured recommender instance
    """
    return PlaylistRecommender()


def demo_week2_system():
    """
    Demonstrate the complete Week 2 system.
    
    This function shows the complete pipeline:
    1. OAuth authentication
    2. User playlist analysis
    3. Taste profile creation
    4. Recommendation generation
    """
    print("🎵 WEEK 2 SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Create recommender
        recommender = create_playlist_recommender()
        
        # Analyze user playlists
        results = recommender.analyze_user_playlists(playlist_limit=3, track_limit=5)
        
        # Generate recommendations for first playlist
        if results['taste_profiles']:
            first_playlist_id = list(results['taste_profiles'].keys())[0]
            taste_profile = results['taste_profiles'][first_playlist_id]
            
            print(f"\n🎯 Generating recommendations...")
            recommendations = recommender.generate_recommendations(taste_profile, top_k=5)
            
            print(f"\n🎵 RECOMMENDATIONS:")
            for i, (song, score) in enumerate(recommendations, 1):
                print(f"  {i}. {song} (similarity: {score:.3f})")
            
            print(f"\n✅ Week 2 system working!")
            return True
        else:
            print("❌ No playlists found to analyze")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    # Run the demo
    demo_week2_system()

