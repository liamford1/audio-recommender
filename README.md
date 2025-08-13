# Audio-Based Music Recommender

A complete music recommendation system with OAuth Spotify integration and proven StandardScaler normalization breakthrough. This system analyzes user playlists and provides personalized music recommendations based on 29-dimensional audio characteristics.

## 🎯 Key Achievements

- ✅ **OAuth Spotify Integration** - Real user playlist access
- ✅ **29-dimensional music feature extraction** - Comprehensive audio analysis
- ✅ **StandardScaler normalization breakthrough** - Critical for realistic similarity scores
- ✅ **Content-based recommendations** - Proven recommendation engine
- ✅ **Complete Week 2 system** - Portfolio-ready demonstration

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete Week 2 system demo
cd demos
python week2_complete_demo.py

# Or try the quick start
python quick_start.py
```

## 🏗️ Project Structure

```
audio-recommender/
├── src/
│   ├── spotify/
│   │   ├── oauth_client.py          # Clean OAuth implementation
│   │   ├── client.py                # Spotify API client
│   │   └── playlist_analyzer.py     # Playlist analysis tools
│   ├── features/
│   │   ├── music_features.py        # 29D feature extraction
│   │   ├── feature_utils.py         # StandardScaler normalization
│   │   └── constants.py             # Feature constants
│   ├── recommender/
│   │   ├── content_based.py         # Core recommendation engine
│   │   └── playlist_recommender.py  # Main Week 2 system
│   └── utils/
│       └── audio_utils.py           # Audio processing utilities
├── demos/
│   ├── week2_complete_demo.py       # Portfolio demonstration
│   └── quick_start.py               # Simple usage example
├── notebooks/                        # Development notebooks
├── tests/
│   └── test_system.py               # System tests
├── data/                            # Audio files and processed data
├── .env                             # Spotify credentials
├── requirements.txt
└── README.md
```

## 🎵 Week 2 System Overview

The complete system demonstrates:

1. **OAuth Authentication** - Secure user authentication with Spotify
2. **Playlist Analysis** - Extract and analyze user playlists
3. **Feature Extraction** - 29-dimensional music feature extraction
4. **StandardScaler Normalization** - The breakthrough that fixed similarity scores
5. **Recommendation Generation** - Content-based recommendations with realistic scores

## 🔧 Core Components

### OAuth Client (`src/spotify/oauth_client.py`)
- **Secure authentication** with Spotify OAuth
- **User playlist access** with proper scopes
- **Error handling** and logging
- **Clean interface** for integration

### Feature Extraction (`src/features/`)
- **29-dimensional features**: chroma, tempo, spectral, MFCC
- **StandardScaler normalization**: Critical breakthrough for realistic similarities
- **Feature validation**: Robust error handling and validation

### Recommendation Engine (`src/recommender/`)
- **Content-based filtering**: Cosine similarity with normalized features
- **Playlist analysis**: Create "taste profiles" from multiple tracks
- **Quality metrics**: Diversity analysis and recommendation statistics

### Main System (`src/recommender/playlist_recommender.py`)
- **Complete pipeline**: OAuth → Playlists → Features → Recommendations
- **Integrated workflow**: All components working together
- **Error handling**: Robust error handling throughout

## 📊 Technical Breakthrough

### StandardScaler Normalization
The critical breakthrough that solved unrealistic similarity scores:

```python
# Before (unrealistic scores)
raw_similarity = 0.98  # Too high due to scale differences

# After (realistic scores)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
realistic_similarity = 0.75  # Proper similarity range
```

This ensures recommendations have meaningful similarity scores in the -0.7 to +0.8 range.

## 🎯 Usage Examples

### Complete Week 2 System
```python
from src.recommender.playlist_recommender import PlaylistRecommender

# Create recommender
recommender = PlaylistRecommender()

# Analyze user playlists and generate recommendations
results = recommender.analyze_user_playlists(playlist_limit=3, track_limit=5)

# Generate recommendations
if results['taste_profiles']:
    taste_profile = list(results['taste_profiles'].values())[0]
    recommendations = recommender.generate_recommendations(taste_profile, top_k=5)
    print("✅ Recommendations generated!")
```

### OAuth Authentication
```python
from src.spotify.oauth_client import create_oauth_client

# Create and authenticate
client = create_oauth_client()
if client.authenticate():
    playlists = client.get_user_playlists(limit=5)
    print(f"Found {len(playlists)} playlists")
```

### Feature Extraction
```python
from src.features.music_features import extract_music_features
from src.features.feature_utils import normalize_features

# Extract features
features = extract_music_features("path/to/song.mp3")

# Normalize (the breakthrough)
normalized_features, scaler = normalize_features(features)
print(f"Extracted and normalized {len(features)} features")
```

## 🧪 Testing

```bash
# Run system tests
python tests/test_system.py

# Test OAuth connection
python src/spotify/oauth_client.py

# Test the complete system
python demos/week2_complete_demo.py
```

## 🔑 Environment Setup

Create a `.env` file with your Spotify credentials:
```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

**Important**: Configure your Spotify app with redirect URI: `http://127.0.0.1:8888`

## 📈 Project Status

🚀 **Week 1 Complete** - Foundation & Core ML ✅
🚀 **Week 2 Complete** - OAuth Integration & Complete System ✅

### Week 1 Achievements
- 29-dimensional music feature extraction
- StandardScaler normalization breakthrough
- Content-based recommendation engine
- Comprehensive testing framework

### Week 2 Achievements
- OAuth Spotify integration
- Real user playlist access
- Complete end-to-end system
- Portfolio-ready demonstrations

## 🎉 Success Criteria Met

After cleanup, the system successfully demonstrates:
```bash
cd demos
python week2_complete_demo.py
# ✅ OAuth user playlist access
# ✅ Feature extraction pipeline  
# ✅ StandardScaler breakthrough preserved
# ✅ Cross-dataset recommendations
# ✅ Realistic similarity scores
```

This demonstrates a complete, production-ready music recommendation system!
