"""
Unit tests for model training module.
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

from models.train_model import load_data, train_models, evaluate_model, save_model
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_features_file():
    """Create temporary features CSV file for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_1': np.random.randn(200),
        'feature_2': np.random.randn(200),
        'feature_3': np.random.randn(200),
        'feature_4': np.random.randn(200),
        'target': np.random.randint(0, 2, 200)
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return Path(f.name)


@pytest.fixture
def sample_train_test_data():
    """Create sample train/test data."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100)
    })
    X_test = pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'feature_3': np.random.randn(50)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    y_test = pd.Series(np.random.randint(0, 2, 50))

    return X_train, X_test, y_train, y_test


def test_load_data_returns_split(sample_features_file):
    """Test that load_data returns train/test split."""
    X_train, X_test, y_train, y_test = load_data(sample_features_file)

    # Check shapes
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Check that test set is smaller
    assert len(X_test) < len(X_train)

    # Cleanup
    sample_features_file.unlink()


def test_load_data_stratified_split(sample_features_file):
    """Test that split maintains class distribution."""
    X_train, X_test, y_train, y_test = load_data(sample_features_file)

    # Check that both classes exist in train and test
    assert len(y_train.unique()) > 1
    assert len(y_test.unique()) > 1

    # Cleanup
    sample_features_file.unlink()


def test_train_models_returns_multiple_models(sample_train_test_data):
    """Test that multiple models are trained."""
    X_train, _, y_train, _ = sample_train_test_data

    trained_models, best_model_name = train_models(X_train, y_train)

    # Check that multiple models are returned
    assert len(trained_models) > 1
    assert 'random_forest' in trained_models
    assert 'logistic_regression' in trained_models

    # Check that best model name is valid
    assert best_model_name in trained_models


def test_train_models_all_fitted(sample_train_test_data):
    """Test that all models are properly fitted."""
    X_train, X_test, y_train, _ = sample_train_test_data

    trained_models, _ = train_models(X_train, y_train)

    # Check that models can make predictions
    for name, model in trained_models.items():
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


def test_evaluate_model_returns_metrics(sample_train_test_data):
    """Test that model evaluation returns proper metrics."""
    X_train, X_test, y_train, y_test = sample_train_test_data

    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Check metrics exist
    assert 'accuracy' in metrics
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics
    assert 'predictions' in metrics

    # Check accuracy is valid
    assert 0 <= metrics['accuracy'] <= 1


def test_evaluate_model_accuracy_range(sample_train_test_data):
    """Test that accuracy is in valid range."""
    X_train, X_test, y_train, y_test = sample_train_test_data

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    # Accuracy should be between 0 and 1
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_save_model_creates_file():
    """Test that model is saved to file."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        save_model(model, 'test_model', output_path)

        # Check file exists
        model_file = output_path / 'test_model_model.pkl'
        assert model_file.exists()

        # Load and verify
        loaded_model = joblib.load(model_file)
        assert isinstance(loaded_model, RandomForestClassifier)


def test_evaluate_model_confusion_matrix_shape(sample_train_test_data):
    """Test that confusion matrix has correct shape."""
    X_train, X_test, y_train, y_test = sample_train_test_data

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    # Confusion matrix should be 2x2 for binary classification
    cm = metrics['confusion_matrix']
    assert cm.shape == (2, 2)


def test_train_models_best_model_selection(sample_train_test_data):
    """Test that best model is selected based on performance."""
    X_train, _, y_train, _ = sample_train_test_data

    trained_models, best_model_name = train_models(X_train, y_train)

    # Best model should be one of the trained models
    assert best_model_name in trained_models
    assert trained_models[best_model_name] is not None
