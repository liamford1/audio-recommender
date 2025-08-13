"""
Spotify OAuth Client Module

This module provides a clean interface for Spotify OAuth authentication
and user playlist access. It consolidates the working OAuth functionality
from the successful authentication tests.
"""

from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SpotifyOAuthClient:
    """
    A clean OAuth client for Spotify API access.
    
    This class handles authentication and provides methods for accessing
    user playlists and track information.
    """
    
    def __init__(self, cache_path: str = ".spotify_cache"):
        """
        Initialize the Spotify OAuth client.
        
        Args:
            cache_path (str): Path to store OAuth cache
        """
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = "http://127.0.0.1:8888"
        self.cache_path = cache_path
        
        if not self.client_id or not self.client_secret:
            raise ValueError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env file")
        
        self.auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="playlist-read-private",
            cache_path=self.cache_path
        )
        
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        logger.info("Spotify OAuth client initialized")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Spotify and verify access.
        
        Returns:
            bool: True if authentication successful
        """
        try:
            user = self.sp.current_user()
            logger.info(f"Successfully authenticated as: {user['display_name']}")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_user_playlists(self, limit: int = 10) -> List[Dict]:
        """
        Get user's playlists.
        
        Args:
            limit (int): Maximum number of playlists to retrieve
            
        Returns:
            List[Dict]: List of playlist objects
        """
        try:
            playlists = self.sp.current_user_playlists(limit=limit)
            logger.info(f"Retrieved {len(playlists['items'])} playlists")
            return playlists['items']
        except Exception as e:
            logger.error(f"Failed to get playlists: {e}")
            return []
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """
        Get tracks from a specific playlist.
        
        Args:
            playlist_id (str): Spotify playlist ID
            
        Returns:
            List[Dict]: List of track objects
        """
        try:
            tracks = self.sp.playlist_tracks(playlist_id)
            logger.info(f"Retrieved {len(tracks['items'])} tracks from playlist")
            return tracks['items']
        except Exception as e:
            logger.error(f"Failed to get playlist tracks: {e}")
            return []
    
    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """
        Get detailed information about a track.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            Dict: Track information or None if failed
        """
        try:
            track = self.sp.track(track_id)
            return track
        except Exception as e:
            logger.error(f"Failed to get track info: {e}")
            return None
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for tracks.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: List of track objects
        """
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            logger.info(f"Found {len(results['tracks']['items'])} tracks for query: {query}")
            return results['tracks']['items']
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


def create_oauth_client() -> SpotifyOAuthClient:
    """
    Factory function to create a Spotify OAuth client.
    
    Returns:
        SpotifyOAuthClient: Configured OAuth client
    """
    return SpotifyOAuthClient()


def test_oauth_connection():
    """
    Test function to verify OAuth connection works.
    
    This function can be used to test the OAuth setup and verify
    that user playlist access is working correctly.
    """
    print("Testing OAuth flow with loopback address...")
    
    try:
        client = create_oauth_client()
        
        if client.authenticate():
            print("✅ OAuth authentication successful!")
            
            # Test playlist access
            playlists = client.get_user_playlists(limit=3)
            print(f"✅ Found {len(playlists)} playlists:")
            
            for playlist in playlists:
                print(f"  - {playlist['name']} ({playlist['tracks']['total']} tracks)")
            
            print("\n🎉 OAuth working! User playlist access successful!")
            return True
        else:
            print("❌ OAuth authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Run OAuth test
    test_oauth_connection()
