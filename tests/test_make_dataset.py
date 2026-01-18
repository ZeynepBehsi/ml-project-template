"""
Unit tests for data processing module.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.make_dataset import load_raw_data, preprocess_data, save_processed_data


@pytest.fixture
def sample_data():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'feature_1': [1, 2, 3, 3, 4],
        'feature_2': ['a', 'b', 'c', 'c', 'd'],
        'feature_3': [1.0, 2.0, np.nan, 3.0, 4.0]
    })


@pytest.fixture
def temp_csv_file(sample_data):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        return Path(f.name)


def test_load_raw_data(temp_csv_file):
    """Test loading raw data from CSV."""
    df = load_raw_data(temp_csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert len(df.columns) == 3
    assert 'feature_1' in df.columns

    # Cleanup
    temp_csv_file.unlink()


def test_preprocess_data_removes_duplicates(sample_data):
    """Test that preprocessing removes duplicate rows."""
    df_processed = preprocess_data(sample_data)

    # Original has 5 rows with 1 duplicate, after dropna should have 3 rows
    assert len(df_processed) < len(sample_data)


def test_preprocess_data_handles_missing_values():
    """Test that preprocessing handles missing values."""
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': [5, 6, 7, 8]
    })

    df_processed = preprocess_data(df)

    # Check no missing values remain
    assert df_processed.isnull().sum().sum() == 0


def test_save_processed_data():
    """Test saving processed data to CSV."""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'output.csv'
        save_processed_data(df, output_path)

        # Check file exists
        assert output_path.exists()

        # Load and verify
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == 3
        assert list(df_loaded.columns) == ['col1', 'col2']


def test_preprocess_data_preserves_columns():
    """Test that preprocessing preserves column names."""
    df = pd.DataFrame({
        'feature_a': [1, 2, 3],
        'feature_b': [4, 5, 6],
        'target': [0, 1, 0]
    })

    df_processed = preprocess_data(df)

    assert set(df_processed.columns) == set(df.columns)
