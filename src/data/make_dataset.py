"""
Script to load and prepare raw data for processing.

Usage:
    python src/data/make_dataset.py --input data/raw/data.csv --output data/processed/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd


def setup_logger():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        input_path: Path to raw data file
        
    Returns:
        DataFrame containing raw data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing of raw data.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Preprocessed dataframe
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing")
    
    # Example preprocessing steps
    # 1. Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()  # or use df.fillna()
    logger.info(f"Handled {missing_before} missing values")
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """
    Save processed data to CSV.
    
    Args:
        df: Processed dataframe
        output_path: Path to save processed data
    """
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path}")


def main():
    """Main execution function."""
    logger = setup_logger()
    
    parser = argparse.ArgumentParser(description='Process raw data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to raw data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save processed data')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) / 'processed_data.csv'
    
    try:
        # Load data
        df = load_raw_data(input_path)
        
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Save
        save_processed_data(df_processed, output_path)
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise


if __name__ == '__main__':
    main()