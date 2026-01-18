"""
Script to train machine learning models.

Usage:
    python src/models/train_model.py --input data/processed/features.csv --output models/
"""

import argparse
import logging
from pathlib import Path
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def setup_logger():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_data(input_path: Path) -> tuple:
    """
    Load feature data and split into train/test sets.

    Args:
        input_path: Path to feature data file

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train multiple models and return the best one.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Dictionary of trained models
    """
    logger = logging.getLogger(__name__)
    logger.info("Training models...")

    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    cv_scores = {}

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")

        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores[name] = cv_score.mean()

        logger.info(f"{name} - CV Accuracy: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")

    # Find best model
    best_model_name = max(cv_scores, key=cv_scores.get)
    logger.info(f"\nBest model: {best_model_name} with CV accuracy: {cv_scores[best_model_name]:.4f}")

    return trained_models, best_model_name


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("\nEvaluating model on test set...")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def save_model(model, model_name: str, output_path: Path):
    """
    Save trained model to disk.

    Args:
        model: Trained model
        model_name: Name of the model
        output_path: Directory to save model
    """
    logger = logging.getLogger(__name__)

    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / f'{model_name}_model.pkl'

    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")


def save_metrics(metrics: dict, output_path: Path):
    """
    Save evaluation metrics to file.

    Args:
        metrics: Dictionary of metrics
        output_path: Directory to save metrics
    """
    logger = logging.getLogger(__name__)

    metrics_file = output_path / 'model_metrics.txt'

    with open(metrics_file, 'w') as f:
        f.write(f"Model Evaluation Metrics\n")
        f.write(f"========================\n\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write(f"Classification Report:\n{metrics['classification_report']}\n")
        f.write(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")

    logger.info(f"Metrics saved to {metrics_file}")


def main():
    """Main execution function."""
    logger = setup_logger()

    parser = argparse.ArgumentParser(description='Train machine learning models')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to feature data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save trained models')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        # Load and split data
        X_train, X_test, y_train, y_test = load_data(input_path)

        # Train models
        trained_models, best_model_name = train_models(X_train, y_train)

        # Get best model
        best_model = trained_models[best_model_name]

        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test)

        # Save model and metrics
        save_model(best_model, best_model_name, output_path)
        save_metrics(metrics, output_path)

        logger.info("\nModel training completed successfully!")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


if __name__ == '__main__':
    main()
