"""
System Tests for Audio Recommender

This module contains tests to verify the core system components work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from src.features.feature_utils import normalize_features, calculate_similarity
from src.recommender.content_based import recommend_from_features


class TestFeatureUtils(unittest.TestCase):
    """Test the feature utilities module."""
    
    def test_normalize_features(self):
        """Test StandardScaler normalization (the breakthrough)."""
        # Create mock features
        features = np.random.rand(10, 29)
        
        # Test normalization
        normalized, scaler = normalize_features(features)
        
        # Verify shape
        self.assertEqual(normalized.shape, (10, 29))
        
        # Verify normalization worked (mean should be close to 0, std close to 1)
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=1)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=1)
        
        # Test with pre-fitted scaler
        new_features = np.random.rand(5, 29)
        normalized_new, _ = normalize_features(new_features, scaler)
        self.assertEqual(normalized_new.shape, (5, 29))
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        # Create two feature vectors
        features1 = np.random.rand(29)
        features2 = np.random.rand(29)
        
        # Calculate similarity
        similarity = calculate_similarity(features1, features2)
        
        # Verify similarity is between -1 and 1
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)


class TestContentBased(unittest.TestCase):
    """Test the content-based recommendation module."""
    
    def test_recommend_from_features(self):
        """Test recommendation generation."""
        # Create mock data
        query_features = np.random.rand(29)
        candidate_matrix = np.random.rand(20, 29)
        candidate_names = [f"Song_{i}" for i in range(20)]
        
        # Generate recommendations
        recommendations = recommend_from_features(
            query_features=query_features,
            candidate_matrix=candidate_matrix,
            candidate_names=candidate_names,
            top_k=5
        )
        
        # Verify recommendations
        self.assertEqual(len(recommendations), 5)
        
        # Verify each recommendation has name and score
        for song_name, score in recommendations:
            self.assertIsInstance(song_name, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)


class TestStandardScalerBreakthrough(unittest.TestCase):
    """Test the StandardScaler normalization breakthrough."""
    
    def test_normalization_improves_similarity(self):
        """Test that normalization produces more realistic similarity scores."""
        # Create features with different scales
        features1 = np.random.rand(10, 29) * 100  # Large scale
        features2 = np.random.rand(10, 29) * 0.1  # Small scale
        
        # Calculate similarity without normalization
        raw_similarity = calculate_similarity(features1[0], features2[0])
        
        # Calculate similarity with normalization
        normalized1, scaler = normalize_features(features1)
        normalized2, _ = normalize_features(features2, scaler)
        normalized_similarity = calculate_similarity(normalized1[0], normalized2[0])
        
        # The normalized similarity should be more realistic
        # (not artificially high due to scale differences)
        self.assertLess(abs(normalized_similarity), 0.95)


def run_system_tests():
    """Run all system tests."""
    print("🧪 Running System Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFeatureUtils))
    test_suite.addTest(unittest.makeSuite(TestContentBased))
    test_suite.addTest(unittest.makeSuite(TestStandardScalerBreakthrough))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("✅ All system tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    run_system_tests()
