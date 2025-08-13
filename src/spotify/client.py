from dotenv import load_dotenv
"""
Spotify API client module.

This module contains the SpotifyClient class for interacting with the Spotify API
to retrieve playlist information and download audio previews.
"""

import os
import re
import requests
from typing import List, Dict, Optional, Tuple
import logging
from urllib.parse import urlparse, parse_qs

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    logging.warning("spotipy not available. Install with: pip install spotipy")

logger = logging.getLogger(__name__)


class SpotifyClient:
    """
    Client for interacting with the Spotify API.
    
    This class handles playlist retrieval and audio preview downloading
    for the music recommendation system.
    """
    
    def __init__(self, client_id: Optional[str] = None, 
                 client_secret: Optional[str] = None):
        """
        Initialize the Spotify client.
        
        Args:
            client_id (str, optional): Spotify client ID. If None, loads from environment
            client_secret (str, optional): Spotify client secret. If None, loads from environment
        """
        # Load environment variables
        load_dotenv()
        
        if not SPOTIPY_AVAILABLE:
            raise ImportError("spotipy is required. Install with: pip install spotipy")
        
        # Load credentials from environment if not provided
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        
        # Initialize Spotipy client
        try:
            self.sp = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            )
            logger.info("Spotify client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise
    
    def extract_playlist_id(self, playlist_url: str) -> str:
        """
        Extract playlist ID from a Spotify playlist URL.
        
        Args:
            playlist_url (str): Spotify playlist URL
            
        Returns:
            str: Playlist ID
            
        Raises:
            ValueError: If URL is not a valid Spotify playlist URL
        """
        # Handle different URL formats
        patterns = [
            r'spotify\.com/playlist/([a-zA-Z0-9]+)',
            r'spotify\.com/user/[^/]+/playlist/([a-zA-Z0-9]+)',
            r'open\.spotify\.com/playlist/([a-zA-Z0-9]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, playlist_url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract playlist ID from URL: {playlist_url}")
    
    def get_playlist_tracks(self, playlist_url: str) -> List[Dict]:
        """
        Get all tracks from a Spotify playlist.
        
        Args:
            playlist_url (str): Spotify playlist URL
            
        Returns:
            List[Dict]: List of track information dictionaries
        """
        try:
            playlist_id = self.extract_playlist_id(playlist_url)
            logger.info(f"Retrieving tracks from playlist: {playlist_id}")
            
            tracks = []
            offset = 0
            limit = 100  # Spotify API limit
            
            while True:
                results = self.sp.playlist_tracks(
                    playlist_id, 
                    offset=offset, 
                    limit=limit,
                    fields='items(track(id,name,artists,preview_url,external_urls))'
                )
                
                if not results['items']:
                    break
                
                for item in results['items']:
                    track = item['track']
                    if track and track['preview_url']:  # Only include tracks with preview URLs
                        track_info = {
                            'id': track['id'],
                            'name': track['name'],
                            'artists': [artist['name'] for artist in track['artists']],
                            'preview_url': track['preview_url'],
                            'spotify_url': track['external_urls']['spotify']
                        }
                        tracks.append(track_info)
                
                offset += limit
                
                if len(results['items']) < limit:
                    break
            
            logger.info(f"Retrieved {len(tracks)} tracks with preview URLs from playlist")
            return tracks
            
        except Exception as e:
            logger.error(f"Error retrieving playlist tracks: {e}")
            raise
    
    def get_playlist_info(self, playlist_url: str) -> Dict:
        """
        Get basic information about a playlist.
        
        Args:
            playlist_url (str): Spotify playlist URL
            
        Returns:
            Dict: Playlist information
        """
        try:
            playlist_id = self.extract_playlist_id(playlist_url)
            
            playlist = self.sp.playlist(
                playlist_id,
                fields='id,name,description,owner,images,tracks.total'
            )
            
            return {
                'id': playlist['id'],
                'name': playlist['name'],
                'description': playlist.get('description', ''),
                'owner': playlist['owner']['display_name'],
                'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                'total_tracks': playlist['tracks']['total']
            }
            
        except Exception as e:
            logger.error(f"Error retrieving playlist info: {e}")
            raise
    
    def download_preview(self, preview_url: str, output_path: str) -> bool:
        """
        Download a 30-second audio preview from Spotify.
        
        Args:
            preview_url (str): Spotify preview URL
            output_path (str): Path to save the audio file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download the audio file
            response = requests.get(preview_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded preview to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading preview {preview_url}: {e}")
            return False
    
    def download_playlist_previews(self, playlist_url: str, 
                                 output_dir: str = "data/processed/playlists") -> List[str]:
        """
        Download all preview URLs from a playlist.
        
        Args:
            playlist_url (str): Spotify playlist URL
            output_dir (str): Directory to save preview files
            
        Returns:
            List[str]: List of paths to downloaded audio files
        """
        tracks = self.get_playlist_tracks(playlist_url)
        downloaded_files = []
        
        for i, track in enumerate(tracks):
            # Create filename from track info
            artist_name = track['artists'][0] if track['artists'] else 'Unknown'
            track_name = track['name']
            
            # Clean filename
            filename = f"{i:03d}_{artist_name}_{track_name}.mp3"
            filename = re.sub(r'[^\w\s-]', '', filename).strip()
            filename = re.sub(r'[-\s]+', '_', filename)
            
            output_path = os.path.join(output_dir, filename)
            
            if self.download_preview(track['preview_url'], output_path):
                downloaded_files.append(output_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} preview files to {output_dir}")
        return downloaded_files
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for tracks on Spotify.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of track information
        """
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            
            tracks = []
            for track in results['tracks']['items']:
                if track['preview_url']:  # Only include tracks with preview URLs
                    track_info = {
                        'id': track['id'],
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']],
                        'preview_url': track['preview_url'],
                        'spotify_url': track['external_urls']['spotify']
                    }
                    tracks.append(track_info)
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            raise
