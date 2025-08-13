"""
Spotify integration module.

This module contains classes for interacting with the Spotify API to analyze
playlists and download audio previews for analysis.
"""

from .client import SpotifyClient
from .playlist_analyzer import PlaylistAnalyzer

__all__ = [
    "SpotifyClient",
    "PlaylistAnalyzer",
]
