"""
Script to build features from processed data.

Usage:
    python src/features/build_features.py --input data/processed/processed_data.csv --output data/processed/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def setup_logger():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load processed data from CSV file.

    Args:
        input_path: Path to processed data file

    Returns:
        DataFrame containing processed data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with new features
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating new features")

    df_features = df.copy()

    # Get feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != 'target']

    # Create interaction features (first 3 features for example)
    if len(feature_cols) >= 3:
        df_features['interaction_0_1'] = df[feature_cols[0]] * df[feature_cols[1]]
        df_features['interaction_0_2'] = df[feature_cols[0]] * df[feature_cols[2]]
        df_features['interaction_1_2'] = df[feature_cols[1]] * df[feature_cols[2]]
        logger.info("Created interaction features")

    # Create polynomial features
    if len(feature_cols) >= 2:
        df_features['poly_0'] = df[feature_cols[0]] ** 2
        df_features['poly_1'] = df[feature_cols[1]] ** 2
        logger.info("Created polynomial features")

    # Create statistical features
    df_features['feature_mean'] = df[feature_cols].mean(axis=1)
    df_features['feature_std'] = df[feature_cols].std(axis=1)
    df_features['feature_max'] = df[feature_cols].max(axis=1)
    df_features['feature_min'] = df[feature_cols].min(axis=1)
    logger.info("Created statistical features")

    logger.info(f"Total features created: {len(df_features.columns) - len(df.columns)}")

    return df_features


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features using StandardScaler.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with scaled features
    """
    logger = logging.getLogger(__name__)
    logger.info("Scaling features")

    # Separate target from features
    target = df['target'].copy()
    feature_cols = [col for col in df.columns if col != 'target']

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    # Create new dataframe
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
    df_scaled['target'] = target.values

    logger.info("Feature scaling completed")

    return df_scaled


def save_features(df: pd.DataFrame, output_path: Path):
    """
    Save feature dataframe to CSV.

    Args:
        df: Feature dataframe
        output_path: Path to save features
    """
    logger = logging.getLogger(__name__)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved features to {output_path}")


def main():
    """Main execution function."""
    logger = setup_logger()

    parser = argparse.ArgumentParser(description='Build features from processed data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to processed data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save feature data')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) / 'features.csv'

    try:
        # Load data
        df = load_data(input_path)

        # Create features
        df_features = create_features(df)

        # Scale features
        df_scaled = scale_features(df_features)

        # Save
        save_features(df_scaled, output_path)

        logger.info("Feature engineering completed successfully!")

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise


if __name__ == '__main__':
    main()
