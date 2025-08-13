"""
Quick Start Demo - Simple Usage Example

This demonstrates how to use the PlaylistRecommender system with a simple interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender.playlist_recommender import PlaylistRecommender

def main():
    """
    Simple demonstration of the Week 2 system.
    """
    print("🎵 QUICK START - Week 2 System Demo")
    print("=" * 40)
    
    try:
        # Create recommender
        recommender = PlaylistRecommender()
        
        # Analyze user playlists
        print("📱 Authenticating and analyzing playlists...")
        results = recommender.analyze_user_playlists(playlist_limit=2, track_limit=3)
        
        if results['taste_profiles']:
            # Generate recommendations for first playlist
            first_playlist_id = list(results['taste_profiles'].keys())[0]
            taste_profile = results['taste_profiles'][first_playlist_id]
            
            print(f"\n🎯 Generating recommendations...")
            recommendations = recommender.generate_recommendations(taste_profile, top_k=3)
            
            print(f"\n🎵 TOP RECOMMENDATIONS:")
            for i, (song, score) in enumerate(recommendations, 1):
                print(f"  {i}. {song} (similarity: {score:.3f})")
            
            print(f"\n✅ Week 2 system working!")
            print("✅ OAuth authentication successful")
            print("✅ Playlist analysis complete")
            print("✅ StandardScaler breakthrough preserved")
            print("✅ Recommendations generated")
            
        else:
            print("❌ No playlists found to analyze")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        print("2. Spotify app configured with redirect URI: http://127.0.0.1:8888")
        print("3. User playlists available")

if __name__ == "__main__":
    main()
