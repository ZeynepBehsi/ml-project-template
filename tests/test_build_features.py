"""
Unit tests for feature engineering module.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.build_features import create_features, scale_features


@pytest.fixture
def sample_data():
    """Create sample dataframe with features and target."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_0': np.random.randn(100),
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })


def test_create_features_adds_new_columns(sample_data):
    """Test that feature engineering creates new features."""
    df_features = create_features(sample_data)

    # Should have more columns than original
    assert len(df_features.columns) > len(sample_data.columns)

    # Original columns should still exist
    for col in sample_data.columns:
        assert col in df_features.columns


def test_create_features_interaction_features(sample_data):
    """Test that interaction features are created."""
    df_features = create_features(sample_data)

    # Check for interaction features
    assert 'interaction_0_1' in df_features.columns
    assert 'interaction_0_2' in df_features.columns
    assert 'interaction_1_2' in df_features.columns


def test_create_features_polynomial_features(sample_data):
    """Test that polynomial features are created."""
    df_features = create_features(sample_data)

    # Check for polynomial features
    assert 'poly_0' in df_features.columns
    assert 'poly_1' in df_features.columns


def test_create_features_statistical_features(sample_data):
    """Test that statistical features are created."""
    df_features = create_features(sample_data)

    # Check for statistical features
    assert 'feature_mean' in df_features.columns
    assert 'feature_std' in df_features.columns
    assert 'feature_max' in df_features.columns
    assert 'feature_min' in df_features.columns


def test_scale_features_preserves_shape(sample_data):
    """Test that scaling preserves dataframe shape."""
    df_scaled = scale_features(sample_data)

    assert df_scaled.shape == sample_data.shape


def test_scale_features_keeps_target(sample_data):
    """Test that scaling keeps the target column."""
    df_scaled = scale_features(sample_data)

    assert 'target' in df_scaled.columns
    # Target should be unchanged
    pd.testing.assert_series_equal(
        df_scaled['target'].reset_index(drop=True),
        sample_data['target'].reset_index(drop=True)
    )


def test_scale_features_standardizes_features(sample_data):
    """Test that features are properly standardized."""
    df_scaled = scale_features(sample_data)

    # Check that scaled features have approximately mean=0 and std=1
    feature_cols = [col for col in df_scaled.columns if col != 'target']

    for col in feature_cols[:3]:  # Check first 3 features
        assert abs(df_scaled[col].mean()) < 1e-10  # Mean should be very close to 0
        assert abs(df_scaled[col].std() - 1.0) < 0.1  # Std should be close to 1


def test_create_features_with_minimal_data():
    """Test feature creation with minimal data."""
    df_min = pd.DataFrame({
        'feature_0': [1, 2, 3],
        'feature_1': [4, 5, 6],
        'target': [0, 1, 0]
    })

    df_features = create_features(df_min)

    # Should still create some features
    assert len(df_features.columns) > len(df_min.columns)


def test_interaction_feature_calculation(sample_data):
    """Test that interaction features are calculated correctly."""
    df_features = create_features(sample_data)

    # Verify interaction calculation
    expected_interaction = sample_data['feature_0'] * sample_data['feature_1']
    pd.testing.assert_series_equal(
        df_features['interaction_0_1'].reset_index(drop=True),
        expected_interaction.reset_index(drop=True),
        check_names=False
    )


def test_polynomial_feature_calculation(sample_data):
    """Test that polynomial features are calculated correctly."""
    df_features = create_features(sample_data)

    # Verify polynomial calculation
    expected_poly = sample_data['feature_0'] ** 2
    pd.testing.assert_series_equal(
        df_features['poly_0'].reset_index(drop=True),
        expected_poly.reset_index(drop=True),
        check_names=False
    )
