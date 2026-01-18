"""
Unit tests for prediction module.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys
import tempfile
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.predict_model import load_model, make_predictions, save_predictions
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def sample_model_file(sample_model):
    """Save model to temporary file."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(sample_model, f.name)
        return Path(f.name)


@pytest.fixture
def sample_features():
    """Create sample features for prediction."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'feature_3': np.random.randn(50)
    })


def test_load_model_returns_model(sample_model_file):
    """Test that load_model loads a valid model."""
    model = load_model(sample_model_file)

    assert model is not None
    assert hasattr(model, 'predict')

    # Cleanup
    sample_model_file.unlink()


def test_load_model_correct_type(sample_model_file):
    """Test that loaded model has correct type."""
    model = load_model(sample_model_file)

    assert isinstance(model, RandomForestClassifier)

    # Cleanup
    sample_model_file.unlink()


def test_make_predictions_returns_array(sample_model, sample_features):
    """Test that make_predictions returns prediction array."""
    predictions, probabilities = make_predictions(sample_model, sample_features)

    assert predictions is not None
    assert len(predictions) == len(sample_features)


def test_make_predictions_binary_values(sample_model, sample_features):
    """Test that predictions are binary (0 or 1)."""
    predictions, _ = make_predictions(sample_model, sample_features)

    # For binary classification, predictions should be 0 or 1
    unique_values = np.unique(predictions)
    assert all(val in [0, 1] for val in unique_values)


def test_make_predictions_with_probabilities(sample_model, sample_features):
    """Test that probabilities are returned."""
    predictions, probabilities = make_predictions(sample_model, sample_features)

    assert probabilities is not None
    assert len(probabilities) == len(sample_features)
    assert probabilities.shape[1] == 2  # Binary classification


def test_make_predictions_probabilities_sum_to_one(sample_model, sample_features):
    """Test that probabilities sum to 1 for each sample."""
    _, probabilities = make_predictions(sample_model, sample_features)

    # Each row should sum to approximately 1
    row_sums = probabilities.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(len(sample_features)))


def test_make_predictions_removes_target_column(sample_model):
    """Test that target column is removed if present."""
    df_with_target = pd.DataFrame({
        'feature_1': np.random.randn(20),
        'feature_2': np.random.randn(20),
        'feature_3': np.random.randn(20),
        'target': np.random.randint(0, 2, 20)
    })

    predictions, _ = make_predictions(sample_model, df_with_target)

    # Should still work and return predictions
    assert len(predictions) == len(df_with_target)


def test_save_predictions_creates_file(sample_model, sample_features):
    """Test that predictions are saved to file."""
    predictions, probabilities = make_predictions(sample_model, sample_features)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        save_predictions(predictions, probabilities, output_path)

        # Check file exists
        predictions_file = output_path / 'predictions.csv'
        assert predictions_file.exists()


def test_save_predictions_file_content(sample_model, sample_features):
    """Test that saved predictions file has correct content."""
    predictions, probabilities = make_predictions(sample_model, sample_features)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        save_predictions(predictions, probabilities, output_path)

        # Load and verify
        df_predictions = pd.read_csv(output_path / 'predictions.csv')

        assert len(df_predictions) == len(predictions)
        assert 'prediction' in df_predictions.columns
        assert 'probability_class_0' in df_predictions.columns
        assert 'probability_class_1' in df_predictions.columns


def test_save_predictions_without_probabilities(sample_model, sample_features):
    """Test saving predictions without probabilities."""
    # Create a model that doesn't have predict_proba
    predictions = np.random.randint(0, 2, len(sample_features))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        save_predictions(predictions, None, output_path)

        # Check file exists
        predictions_file = output_path / 'predictions.csv'
        assert predictions_file.exists()

        # Load and verify
        df_predictions = pd.read_csv(predictions_file)
        assert 'prediction' in df_predictions.columns


def test_predictions_match_model_output(sample_model, sample_features):
    """Test that predictions match direct model output."""
    predictions, _ = make_predictions(sample_model, sample_features)

    # Direct prediction from model
    direct_predictions = sample_model.predict(sample_features)

    np.testing.assert_array_equal(predictions, direct_predictions)


def test_make_predictions_consistent_shape(sample_model):
    """Test that predictions have consistent shape with input."""
    for n_samples in [10, 50, 100]:
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples)
        })

        predictions, probabilities = make_predictions(sample_model, df)

        assert len(predictions) == n_samples
        assert len(probabilities) == n_samples
