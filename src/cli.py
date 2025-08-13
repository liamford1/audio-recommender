"""
Command-line interface for the music recommender system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.features.music_features import extract_music_features
from src.features.feature_utils import normalize_features, load_feature_components
from src.recommender.content_based import recommend_with_normalized_features, print_recommendations
from src.spotify.playlist_analyzer import PlaylistAnalyzer


def test_setup():
    """Test that the basic setup is working."""
    print("🧪 Testing Music Recommender Setup")
    print("=" * 40)
    
    try:
        # Test feature extraction
        print("✅ Feature extraction module imported")
        
        # Test recommendation module
        print("✅ Recommendation module imported")
        
        # Test Spotify module
        print("✅ Spotify integration module imported")
        
        print("\n🎉 All modules loaded successfully!")
        print("The music recommender system is ready to use.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check that all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def extract_features(audio_file):
    """Extract features from a single audio file."""
    print(f"🎵 Extracting features from: {audio_file}")
    
    try:
        features = extract_music_features(audio_file)
        print(f"✅ Extracted {len(features)} features")
        print(f"Feature vector: {features[:5]}...")  # Show first 5 features
        return features
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None


def load_and_test_recommendations():
    """Load saved components and test recommendations."""
    print("\n🎯 Testing Recommendations with Saved Data")
    print("=" * 50)
    
    try:
        # Try to load saved components
        raw_matrix, normalized_matrix, scaler, track_info = load_feature_components(
            'data/processed/embeddings'
        )
        
        print(f"✅ Loaded {len(track_info)} tracks")
        print(f"Matrix shape: {normalized_matrix.shape}")
        
        # Test recommendations for first track
        if len(track_info) > 0:
            recommendations = recommend_with_normalized_features(
                query_idx=0,
                normalized_matrix=normalized_matrix,
                track_names=track_info,
                top_k=3
            )
            
            print_recommendations(track_info[0], recommendations)
        
        return True
        
    except Exception as e:
        print(f"❌ Could not load saved data: {e}")
        print("This is expected if you haven't run the notebooks yet.")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Music Recommender System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli test                    # Test the setup
  python -m src.cli extract song.mp3        # Extract features from a song
  python -m src.cli recommend               # Test recommendations
        """
    )
    
    parser.add_argument(
        'command',
        choices=['test', 'extract', 'recommend'],
        help='Command to run'
    )
    
    parser.add_argument(
        'audio_file',
        nargs='?',
        help='Audio file path (for extract command)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'test':
        success = test_setup()
        sys.exit(0 if success else 1)
    
    elif args.command == 'extract':
        if not args.audio_file:
            print("❌ Please provide an audio file path")
            sys.exit(1)
        
        if not os.path.exists(args.audio_file):
            print(f"❌ Audio file not found: {args.audio_file}")
            sys.exit(1)
        
        features = extract_features(args.audio_file)
        sys.exit(0 if features is not None else 1)
    
    elif args.command == 'recommend':
        success = load_and_test_recommendations()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
