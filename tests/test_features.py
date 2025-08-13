"""
Tests for music feature extraction and recommendation functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the modules to test
from src.features.music_features import extract_music_features
from src.features.constants import get_feature_names
from src.features.feature_utils import (
    normalize_features, 
    calculate_similarity, 
    get_similarity_statistics,
    validate_feature_matrix
)
from src.recommender.content_based import (
    recommend_with_normalized_features,
    recommend_from_features,
    get_recommendation_quality_stats
)


class TestMusicFeatures(unittest.TestCase):
    """Test music feature extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock audio file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.mock_audio_file = os.path.join(self.temp_dir, "test_audio.mp3")
        
        # Create a dummy file
        with open(self.mock_audio_file, 'w') as f:
            f.write("dummy audio content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_feature_names(self):
        """Test that feature names are correctly defined."""
        feature_names = get_feature_names()
        
        # Should have exactly 29 features
        self.assertEqual(len(feature_names), 29)
        
        # Should have the expected feature types
        self.assertTrue(any('chroma_' in name for name in feature_names[:12]))
        self.assertEqual(feature_names[12], 'tempo')
        self.assertTrue('spectral_centroid' in feature_names[13:16])
        self.assertTrue('mfcc_' in feature_names[16])
    
    @patch('librosa.load')
    @patch('librosa.feature.chroma_stft')
    @patch('librosa.beat.beat_track')
    @patch('librosa.feature.spectral_centroid')
    @patch('librosa.feature.spectral_rolloff')
    @patch('librosa.feature.zero_crossing_rate')
    @patch('librosa.feature.mfcc')
    def test_extract_music_features(self, mock_mfcc, mock_zcr, mock_rolloff, 
                                   mock_centroid, mock_beat, mock_chroma, mock_load):
        """Test music feature extraction with mocked librosa."""
        # Mock librosa functions
        mock_load.return_value = (np.random.rand(22050 * 30), 22050)  # 30 seconds at 22.05kHz
        
        # Mock chroma features (12 values)
        mock_chroma.return_value = np.random.rand(12, 100)
        
        # Mock tempo
        mock_beat.return_value = (120.0, np.array([0, 100, 200]))
        
        # Mock spectral features
        mock_centroid.return_value = np.array([[0.5]])
        mock_rolloff.return_value = np.array([[0.7]])
        mock_zcr.return_value = np.array([[0.1]])
        
        # Mock MFCC features (13 values)
        mock_mfcc.return_value = np.random.rand(13, 100)
        
        # Extract features
        features = extract_music_features(self.mock_audio_file)
        
        # Check output
        self.assertEqual(len(features), 29)
        self.assertIsInstance(features, np.ndarray)
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))
    
    def test_extract_music_features_file_not_found(self):
        """Test that extract_music_features raises error for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            extract_music_features("non_existent_file.mp3")


class TestFeatureUtils(unittest.TestCase):
    """Test feature utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample feature matrix
        self.feature_matrix = np.random.rand(10, 29)  # 10 songs, 29 features
        self.track_names = [f"song_{i}" for i in range(10)]
    
    def test_normalize_features(self):
        """Test feature normalization."""
        normalized_matrix, scaler = normalize_features(self.feature_matrix)
        
        # Check output shapes
        self.assertEqual(normalized_matrix.shape, self.feature_matrix.shape)
        
        # Check that normalized features have mean ~0 and std ~1
        self.assertAlmostEqual(np.mean(normalized_matrix), 0.0, places=1)
        self.assertAlmostEqual(np.std(normalized_matrix), 1.0, places=1)
        
        # Check that scaler can be reused
        new_features = np.random.rand(5, 29)
        new_normalized, _ = normalize_features(new_features, scaler)
        self.assertEqual(new_normalized.shape, (5, 29))
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        normalized_matrix, _ = normalize_features(self.feature_matrix)
        similarities = calculate_similarity(0, feature_matrix=normalized_matrix)
        
        # Check output
        self.assertEqual(len(similarities), 10)
        self.assertAlmostEqual(similarities[0], 1.0)  # Self-similarity should be 1
        self.assertTrue(all(-1 <= s <= 1 for s in similarities))  # Cosine similarity bounds
    
    def test_get_similarity_statistics(self):
        """Test similarity statistics calculation."""
        # Create a similarity matrix
        similarity_matrix = np.random.rand(5, 5)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(similarity_matrix, 1.0)  # Diagonal should be 1
        
        stats = get_similarity_statistics(similarity_matrix)
        
        # Check that all expected keys are present
        expected_keys = ['mean', 'std', 'min', 'max', 'median']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))
    
    def test_validate_feature_matrix(self):
        """Test feature matrix validation."""
        # Valid matrix
        self.assertTrue(validate_feature_matrix(self.feature_matrix))
        
        # Invalid dimensions
        invalid_matrix = np.random.rand(10, 30)  # Wrong number of features
        self.assertFalse(validate_feature_matrix(invalid_matrix))
        
        # Contains NaN
        nan_matrix = self.feature_matrix.copy()
        nan_matrix[0, 0] = np.nan
        self.assertFalse(validate_feature_matrix(nan_matrix))
        
        # Contains Inf
        inf_matrix = self.feature_matrix.copy()
        inf_matrix[0, 0] = np.inf
        self.assertFalse(validate_feature_matrix(inf_matrix))


class TestContentBasedRecommender(unittest.TestCase):
    """Test content-based recommendation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.feature_matrix = np.random.rand(10, 29)
        self.normalized_matrix, self.scaler = normalize_features(self.feature_matrix)
        self.track_names = [f"song_{i}" for i in range(10)]
    
    def test_recommend_with_normalized_features(self):
        """Test recommendation with normalized features."""
        recommendations = recommend_with_normalized_features(
            query_idx=0,
            normalized_matrix=self.normalized_matrix,
            track_names=self.track_names,
            top_k=3
        )
        
        # Check output format
        self.assertEqual(len(recommendations), 3)
        self.assertIsInstance(recommendations[0], tuple)
        self.assertEqual(len(recommendations[0]), 2)
        
        # Check that query song is not in recommendations
        query_song = self.track_names[0]
        recommended_songs = [rec[0] for rec in recommendations]
        self.assertNotIn(query_song, recommended_songs)
        
        # Check similarity scores are in valid range
        for _, similarity in recommendations:
            self.assertTrue(-1 <= similarity <= 1)
    
    def test_recommend_from_features(self):
        """Test recommendation from feature vector."""
        query_features = np.random.rand(29)
        
        recommendations = recommend_from_features(
            query_features=query_features,
            candidate_matrix=self.normalized_matrix,
            candidate_names=self.track_names,
            top_k=3
        )
        
        # Check output
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(all(-1 <= s <= 1 for _, s in recommendations))
    
    def test_recommendation_quality_stats(self):
        """Test recommendation quality statistics."""
        recommendations = [
            ("song_1", 0.8),
            ("song_2", 0.6),
            ("song_3", -0.2),
            ("song_4", 0.4)
        ]
        
        stats = get_recommendation_quality_stats(recommendations)
        
        # Check expected keys
        expected_keys = ['count', 'mean_similarity', 'max_similarity', 
                        'min_similarity', 'std_similarity', 'positive_count', 'negative_count']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check values
        self.assertEqual(stats['count'], 4)
        self.assertEqual(stats['positive_count'], 3)
        self.assertEqual(stats['negative_count'], 1)
        self.assertEqual(stats['max_similarity'], 0.8)
        self.assertEqual(stats['min_similarity'], -0.2)
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        # Invalid query index
        with self.assertRaises(ValueError):
            recommend_with_normalized_features(
                query_idx=20,  # Out of bounds
                normalized_matrix=self.normalized_matrix,
                track_names=self.track_names
            )
        
        # Mismatched dimensions
        with self.assertRaises(ValueError):
            recommend_with_normalized_features(
                query_idx=0,
                normalized_matrix=self.normalized_matrix,
                track_names=self.track_names[:5]  # Wrong length
            )
        
        # Invalid feature dimensions
        with self.assertRaises(ValueError):
            recommend_from_features(
                query_features=np.random.rand(30),  # Wrong size
                candidate_matrix=self.normalized_matrix,
                candidate_names=self.track_names
            )


if __name__ == '__main__':
    unittest.main()
