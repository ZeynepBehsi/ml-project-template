"""
Script to make predictions using trained models.

Usage:
    python src/models/predict_model.py --model models/random_forest_model.pkl --input data/processed/features.csv --output reports/
"""

import argparse
import logging
from pathlib import Path
import joblib

import pandas as pd
import numpy as np


def setup_logger():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model(model_path: Path):
    """
    Load trained model from disk.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path}")

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")

    return model


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load data for prediction.

    Args:
        input_path: Path to input data file

    Returns:
        DataFrame with features
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def make_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the model.

    Args:
        model: Trained model
        X: Features for prediction

    Returns:
        Array of predictions
    """
    logger = logging.getLogger(__name__)
    logger.info("Making predictions...")

    # Remove target column if present
    if 'target' in X.columns:
        X = X.drop('target', axis=1)

    predictions = model.predict(X)
    logger.info(f"Generated {len(predictions)} predictions")

    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        logger.info("Prediction probabilities calculated")
        return predictions, probabilities

    return predictions, None


def save_predictions(predictions: np.ndarray, probabilities: np.ndarray,
                    output_path: Path):
    """
    Save predictions to CSV file.

    Args:
        predictions: Array of predictions
        probabilities: Array of prediction probabilities
        output_path: Path to save predictions
    """
    logger = logging.getLogger(__name__)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create predictions dataframe
    df_predictions = pd.DataFrame({
        'prediction': predictions
    })

    # Add probabilities if available
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            df_predictions[f'probability_class_{i}'] = probabilities[:, i]

    # Save to CSV
    predictions_file = output_path / 'predictions.csv'
    df_predictions.to_csv(predictions_file, index=False)

    logger.info(f"Predictions saved to {predictions_file}")

    # Print summary
    logger.info("\nPrediction Summary:")
    logger.info(f"Total predictions: {len(predictions)}")
    unique, counts = np.unique(predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count} ({count/len(predictions)*100:.2f}%)")


def main():
    """Main execution function."""
    logger = setup_logger()

    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save predictions')

    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        # Load model
        model = load_model(model_path)

        # Load data
        df = load_data(input_path)

        # Make predictions
        predictions, probabilities = make_predictions(model, df)

        # Save predictions
        save_predictions(predictions, probabilities, output_path)

        logger.info("\nPrediction completed successfully!")

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


if __name__ == '__main__':
    main()
