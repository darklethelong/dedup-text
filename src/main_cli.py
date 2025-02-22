import argparse
import yaml
import pandas as pd
import plotly.io as pio
from pathlib import Path
import logging

from data_loader import DataLoader
from text_processor import TextProcessor
from visualizer import Visualizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_plot(fig, filename: str, output_dir: Path):
    """Save plot to HTML format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{filename}.html"
    
    # Save as interactive HTML
    fig.write_html(str(html_path))

def main():
    parser = argparse.ArgumentParser(description='Text Deduplication Tool')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize components
        data_loader = DataLoader(config)
        text_processor = TextProcessor(config)
        visualizer = Visualizer(config)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and process data
        logger.info("Loading data...")
        df = pd.read_csv(args.input)
        data_loader.validate_data(df)

        logger.info("Processing texts...")
        deduplicated_df, dedup_info, similarity_matrix, embeddings, labels = text_processor.process(df)

        # Save deduplicated data
        output_csv = output_dir / "deduplicated_data.csv"
        deduplicated_df.to_csv(output_csv, index=False)
        logger.info(f"Saved deduplicated data to {output_csv}")

        # Generate and save visualizations
        logger.info("Generating visualizations...")
        
        # Text Length Distribution
        fig = visualizer.plot_text_length_distribution(df, "Original Text Length Distribution")
        save_plot(fig, "text_length_dist_original", output_dir)
        
        fig = visualizer.plot_text_length_distribution(deduplicated_df, "Deduplicated Text Length Distribution")
        save_plot(fig, "text_length_dist_dedup", output_dir)

        # Similarity Heatmap
        fig = visualizer.plot_similarity_heatmap(
            similarity_matrix,
            df[config['data']['text_column']].tolist()
        )
        save_plot(fig, "similarity_heatmap", output_dir)

        # Cluster Visualization
        fig = visualizer.plot_cluster_visualization(
            embeddings,
            labels,
            df[config['data']['text_column']].tolist()
        )
        save_plot(fig, "cluster_visualization", output_dir)

        # Deduplication Summary
        fig = visualizer.plot_deduplication_summary(dedup_info)
        save_plot(fig, "deduplication_summary", output_dir)

        # Print summary statistics
        print("\nDeduplication Summary:")
        print(f"Original texts: {len(df)}")
        print(f"Unique texts: {len(deduplicated_df)}")
        print(f"Duplicates removed: {dedup_info['duplicates_removed']}")
        print(f"Number of clusters: {dedup_info['total_clusters']}")
        print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 