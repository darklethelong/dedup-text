import argparse
import yaml
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
import numpy as np

from data_loader import DataLoader
from text_processor import TextProcessor
from visualizer import Visualizer

def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Text Deduplication Tool for Call Center Utterances'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file (overrides config file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file (overrides config file)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Similarity threshold (overrides config file)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Launch Streamlit visualization dashboard'
    )
    return parser

def process_data(
    config: dict,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Main data processing pipeline."""
    # Override config with command line arguments
    if input_file:
        config['data']['input_file'] = input_file
    if output_file:
        config['data']['output_file'] = output_file
    if threshold:
        config['deduplication']['initial_threshold'] = threshold

    # Initialize components
    data_loader = DataLoader(config)
    text_processor = TextProcessor(config)
    
    # Load data
    df, error = data_loader.load_data()
    if error:
        raise ValueError(error)
        
    # Process data
    deduplicated_df, dedup_info, similarity_matrix, embeddings, labels = text_processor.process(df)
    
    # Save results
    data_loader.save_data(deduplicated_df)
    
    return df, deduplicated_df, dedup_info, similarity_matrix, embeddings, labels

def run_visualization(
    config: dict,
    original_df: pd.DataFrame,
    deduplicated_df: pd.DataFrame,
    similarity_matrix: Optional[pd.DataFrame] = None,
    embeddings: Optional[pd.DataFrame] = None,
    labels: Optional[pd.DataFrame] = None,
    dedup_info: Optional[dict] = None
) -> None:
    """Run the Streamlit visualization dashboard."""
    visualizer = Visualizer(config)
    visualizer.create_streamlit_dashboard(
        original_df,
        deduplicated_df,
        similarity_matrix,
        embeddings,
        labels,
        dedup_info
    )

def main():
    """Main entry point."""
    try:
        # Parse command line arguments
        parser = setup_argparse()
        args = parser.parse_args()
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Process data
        original_df, deduplicated_df, dedup_info, similarity_matrix, embeddings, labels = process_data(
            config,
            args.input,
            args.output,
            args.threshold
        )
        
        # Print summary
        print("\nDeduplication Summary:")
        print(f"Original texts: {len(original_df)}")
        print(f"Unique texts: {len(deduplicated_df)}")
        print(f"Duplicates removed: {dedup_info['duplicates_removed']}")
        print(f"Number of clusters: {dedup_info['total_clusters']}")
        
        # Launch visualization if requested
        if args.visualize:
            run_visualization(
                config,
                original_df,
                deduplicated_df,
                similarity_matrix,
                embeddings,
                labels,
                dedup_info
            )
            
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 