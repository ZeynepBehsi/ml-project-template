"""
Script to create visualizations from data and model results.

Usage:
    python src/visualization/visualize.py --data data/processed/features.csv --predictions reports/predictions.csv --output reports/figures/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def setup_logger():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_data(data_path: Path, predictions_path: Path = None) -> tuple:
    """
    Load data and predictions.

    Args:
        data_path: Path to feature data
        predictions_path: Path to predictions file (optional)

    Returns:
        Tuple of (data_df, predictions_df)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {data_path}")

    data_df = pd.read_csv(data_path)

    predictions_df = None
    if predictions_path and predictions_path.exists():
        logger.info(f"Loading predictions from {predictions_path}")
        predictions_df = pd.read_csv(predictions_path)

    return data_df, predictions_df


def plot_feature_distributions(df: pd.DataFrame, output_path: Path):
    """
    Plot distributions of features.

    Args:
        df: Feature dataframe
        output_path: Path to save plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating feature distribution plots...")

    # Select first 6 features for visualization
    feature_cols = [col for col in df.columns if col != 'target'][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / 'feature_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature distributions to {output_file}")


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path):
    """
    Plot correlation matrix of features.

    Args:
        df: Feature dataframe
        output_path: Path to save plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating correlation matrix...")

    # Calculate correlation matrix (limit to first 15 features)
    feature_cols = [col for col in df.columns if col != 'target'][:15]
    corr_matrix = df[feature_cols].corr()

    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()

    output_file = output_path / 'correlation_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved correlation matrix to {output_file}")


def plot_target_distribution(df: pd.DataFrame, output_path: Path):
    """
    Plot distribution of target variable.

    Args:
        df: Dataframe with target
        output_path: Path to save plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating target distribution plot...")

    if 'target' not in df.columns:
        logger.warning("No target column found, skipping target distribution plot")
        return

    # Count plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    target_counts = df['target'].value_counts().sort_index()
    axes[0].bar(target_counts.index, target_counts.values, alpha=0.7, edgecolor='black')
    axes[0].set_title('Target Distribution (Count)', fontsize=14)
    axes[0].set_xlabel('Target Class')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Pie chart
    axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['skyblue', 'lightcoral'])
    axes[1].set_title('Target Distribution (Percentage)', fontsize=14)

    plt.tight_layout()
    output_file = output_path / 'target_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved target distribution to {output_file}")


def plot_prediction_distribution(predictions_df: pd.DataFrame, output_path: Path):
    """
    Plot distribution of predictions.

    Args:
        predictions_df: Predictions dataframe
        output_path: Path to save plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating prediction distribution plot...")

    if predictions_df is None or 'prediction' not in predictions_df.columns:
        logger.warning("No predictions found, skipping prediction distribution plot")
        return

    # Count plot
    fig, ax = plt.subplots(figsize=(10, 6))

    pred_counts = predictions_df['prediction'].value_counts().sort_index()
    ax.bar(pred_counts.index, pred_counts.values, alpha=0.7, edgecolor='black', color='green')
    ax.set_title('Prediction Distribution', fontsize=14)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(pred_counts.values):
        ax.text(pred_counts.index[i], v + 1, str(v), ha='center', va='bottom')

    plt.tight_layout()
    output_file = output_path / 'prediction_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved prediction distribution to {output_file}")


def plot_feature_importance(data_df: pd.DataFrame, output_path: Path):
    """
    Plot feature importance based on variance.

    Args:
        data_df: Feature dataframe
        output_path: Path to save plot
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating feature importance plot...")

    # Calculate variance for each feature
    feature_cols = [col for col in data_df.columns if col != 'target']
    variances = data_df[feature_cols].var().sort_values(ascending=False)[:15]

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(variances)), variances.values, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(variances)), variances.index)
    plt.xlabel('Variance')
    plt.title('Top 15 Features by Variance', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    output_file = output_path / 'feature_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature importance to {output_file}")


def main():
    """Main execution function."""
    logger = setup_logger()

    parser = argparse.ArgumentParser(description='Create visualizations')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to feature data file')
    parser.add_argument('--predictions', type=str, default=None,
                       help='Path to predictions file (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save visualizations')

    args = parser.parse_args()

    data_path = Path(args.data)
    predictions_path = Path(args.predictions) if args.predictions else None
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        data_df, predictions_df = load_data(data_path, predictions_path)

        # Create visualizations
        plot_feature_distributions(data_df, output_path)
        plot_correlation_matrix(data_df, output_path)
        plot_target_distribution(data_df, output_path)
        plot_feature_importance(data_df, output_path)

        if predictions_df is not None:
            plot_prediction_distribution(predictions_df, output_path)

        logger.info("\nVisualization completed successfully!")
        logger.info(f"All plots saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise


if __name__ == '__main__':
    main()
