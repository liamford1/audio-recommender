"""
Playlist analyzer module.

This module contains the PlaylistAnalyzer class that combines Spotify playlist
retrieval with the music recommendation engine to provide playlist-based recommendations.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from ..spotify.client import SpotifyClient
from ..features.music_features import extract_music_features
from ..features.feature_utils import normalize_features
# Removed circular import - PlaylistRecommender will be imported when needed

logger = logging.getLogger(__name__)


class PlaylistAnalyzer:
    """
    Analyzes Spotify playlists and provides recommendations.
    
    This class combines Spotify API integration with the music recommendation
    engine to analyze playlists and generate personalized recommendations.
    """
    
    def __init__(self, spotify_client: Optional[SpotifyClient] = None,
                 candidate_matrix: Optional[np.ndarray] = None,
                 candidate_names: Optional[List[str]] = None,
                 scaler: Optional[object] = None):
        """
        Initialize the playlist analyzer.
        
        Args:
            spotify_client (SpotifyClient, optional): Initialized Spotify client
            candidate_matrix (np.ndarray, optional): Pre-computed candidate feature matrix
            candidate_names (List[str], optional): Names of candidate songs
            scaler (object, optional): Pre-fitted StandardScaler
        """
        self.spotify_client = spotify_client
        # Import locally to avoid circular import
        from ..recommender.playlist_recommender import PlaylistRecommender
        self.recommender = PlaylistRecommender(
            candidate_matrix=candidate_matrix,
            candidate_names=candidate_names,
            scaler=scaler
        )
        
        logger.info("Initialized PlaylistAnalyzer")
    
    def set_spotify_client(self, spotify_client: SpotifyClient) -> None:
        """
        Set the Spotify client for playlist analysis.
        
        Args:
            spotify_client (SpotifyClient): Initialized Spotify client
        """
        self.spotify_client = spotify_client
        logger.info("Spotify client set for playlist analysis")
    
    def analyze_playlist_from_url(self, playlist_url: str, 
                                 download_previews: bool = True,
                                 output_dir: str = "data/processed/playlists") -> Dict:
        """
        Analyze a Spotify playlist from URL and generate recommendations.
        
        Args:
            playlist_url (str): Spotify playlist URL
            download_previews (bool): Whether to download audio previews
            output_dir (str): Directory to save downloaded previews
            
        Returns:
            Dict: Analysis results including recommendations
        """
        if not self.spotify_client:
            raise ValueError("Spotify client not set. Use set_spotify_client() first.")
        
        try:
            # Get playlist information
            playlist_info = self.spotify_client.get_playlist_info(playlist_url)
            logger.info(f"Analyzing playlist: {playlist_info['name']}")
            
            # Get playlist tracks
            tracks = self.spotify_client.get_playlist_tracks(playlist_url)
            
            if not tracks:
                raise ValueError("No tracks with preview URLs found in playlist")
            
            # Download previews if requested
            audio_files = []
            if download_previews:
                audio_files = self.spotify_client.download_playlist_previews(
                    playlist_url, output_dir
                )
            
            # Analyze the playlist
            analysis_results = self._analyze_playlist_tracks(
                tracks, audio_files, playlist_info
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing playlist: {e}")
            raise
    
    def _analyze_playlist_tracks(self, tracks: List[Dict], 
                                audio_files: List[str],
                                playlist_info: Dict) -> Dict:
        """
        Analyze playlist tracks and generate recommendations.
        
        Args:
            tracks (List[Dict]): List of track information from Spotify
            audio_files (List[str]): List of paths to downloaded audio files
            playlist_info (Dict): Playlist information
            
        Returns:
            Dict: Analysis results
        """
        # Extract features from audio files
        track_features = []
        successful_tracks = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                features = extract_music_features(audio_file)
                track_features.append(features)
                successful_tracks.append(tracks[i])
                logger.info(f"Extracted features from {os.path.basename(audio_file)}")
            except Exception as e:
                logger.warning(f"Failed to extract features from {audio_file}: {e}")
                continue
        
        if not track_features:
            raise ValueError("No valid audio features extracted from playlist")
        
        # Create playlist profile
        playlist_profile = self.recommender.create_playlist_profile(track_features)
        
        # Calculate playlist diversity
        diversity_stats = self.recommender.get_playlist_diversity(track_features)
        
        # Generate recommendations if we have candidates
        recommendations = []
        if self.recommender.candidate_matrix is not None:
            recommendations = self.recommender.recommend_from_playlist(
                playlist_profile, top_k=10
            )
        
        # Compile results
        results = {
            'playlist_info': playlist_info,
            'track_count': len(tracks),
            'successful_tracks': len(successful_tracks),
            'playlist_profile': playlist_profile,
            'diversity_stats': diversity_stats,
            'recommendations': recommendations,
            'audio_files': audio_files,
            'track_features': track_features
        }
        
        logger.info(f"Playlist analysis complete: {len(successful_tracks)}/{len(tracks)} tracks processed")
        return results
    
    def create_playlist_profile(self, tracks: List[Dict]) -> np.ndarray:
        """
        Create a playlist profile from track information.
        
        This method downloads previews and extracts features to create
        a musical profile of the playlist.
        
        Args:
            tracks (List[Dict]): List of track information from Spotify
            
        Returns:
            np.ndarray: 29-dimensional playlist profile
        """
        if not self.spotify_client:
            raise ValueError("Spotify client not set")
        
        # Download previews for all tracks
        audio_files = []
        for i, track in enumerate(tracks):
            if track['preview_url']:
                # Create temporary filename
                filename = f"temp_{i:03d}_{track['name']}.mp3"
                filename = filename.replace(' ', '_').replace('/', '_')
                temp_path = f"data/processed/playlists/{filename}"
                
                if self.spotify_client.download_preview(track['preview_url'], temp_path):
                    audio_files.append(temp_path)
        
        if not audio_files:
            raise ValueError("No audio previews could be downloaded")
        
        # Extract features
        track_features = []
        for audio_file in audio_files:
            try:
                features = extract_music_features(audio_file)
                track_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features from {audio_file}: {e}")
                continue
        
        if not track_features:
            raise ValueError("No features could be extracted from audio files")
        
        # Create playlist profile
        playlist_profile = self.recommender.create_playlist_profile(track_features)
        
        # Clean up temporary files
        for audio_file in audio_files:
            try:
                os.remove(audio_file)
            except:
                pass
        
        return playlist_profile
    
    def recommend_from_playlist(self, playlist_url: str, 
                               candidate_songs: Optional[List[str]] = None,
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations from a Spotify playlist.
        
        Args:
            playlist_url (str): Spotify playlist URL
            candidate_songs (List[str], optional): List of candidate song paths
            top_k (int): Number of recommendations to return
            
        Returns:
            List[Tuple[str, float]]: List of recommendations
        """
        if not self.spotify_client:
            raise ValueError("Spotify client not set")
        
        # Analyze the playlist
        analysis = self.analyze_playlist_from_url(playlist_url)
        playlist_profile = analysis['playlist_profile']
        
        # If candidate songs provided, analyze them
        if candidate_songs:
            candidate_features = []
            candidate_names = []
            
            for song_path in candidate_songs:
                try:
                    features = extract_music_features(song_path)
                    candidate_features.append(features)
                    candidate_names.append(os.path.basename(song_path))
                except Exception as e:
                    logger.warning(f"Failed to extract features from {song_path}: {e}")
                    continue
            
            if candidate_features:
                # Normalize candidate features
                candidate_matrix = np.array(candidate_features)
                normalized_matrix, scaler = normalize_features(candidate_matrix)
                
                # Generate recommendations
                recommendations = self.recommender.recommend_from_playlist(
                    playlist_profile, top_k=top_k
                )
                
                return recommendations
        
        # Use existing candidate matrix if available
        if self.recommender.candidate_matrix is not None:
            return self.recommender.recommend_from_playlist(playlist_profile, top_k=top_k)
        
        return []
    
    def print_playlist_analysis(self, analysis_results: Dict) -> None:
        """
        Print a comprehensive analysis of playlist results.
        
        Args:
            analysis_results (Dict): Results from analyze_playlist_from_url
        """
        playlist_info = analysis_results['playlist_info']
        diversity_stats = analysis_results['diversity_stats']
        recommendations = analysis_results['recommendations']
        
        print(f"🎵 Playlist Analysis: {playlist_info['name']}")
        print("=" * 60)
        print(f"📊 Playlist Statistics:")
        print(f"   - Owner: {playlist_info['owner']}")
        print(f"   - Total Tracks: {analysis_results['track_count']}")
        print(f"   - Processed Tracks: {analysis_results['successful_tracks']}")
        print(f"   - Diversity Score: {diversity_stats['diversity_score']:.3f}")
        print(f"   - Feature Variance: {diversity_stats['feature_variance']:.3f}")
        print()
        
        if recommendations:
            print("🎯 Top Recommendations:")
            for i, (track_name, similarity_score) in enumerate(recommendations, 1):
                print(f"   {i:2d}. {track_name} (similarity: {similarity_score:.3f})")
        else:
            print("⚠️  No recommendations available (no candidate matrix loaded)")
        print()
