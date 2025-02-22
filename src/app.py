# Import configuration first
import streamlit_config

# Standard library imports
import os
import sys
import yaml
from pathlib import Path

# Ensure proper Python path
sys.path.append(str(Path(__file__).parent.parent))

# Third-party imports
import streamlit as st
import pandas as pd

# Local imports
from data_loader import DataLoader
from text_processor import TextProcessor
from visualizer import Visualizer

def main():
    # Configure Streamlit
    st.set_page_config(
        page_title="Text Deduplication Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Text Deduplication Analysis Dashboard")
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with 'texts' and 'labels' columns"
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Initialize components
        data_loader = DataLoader(config)
        text_processor = TextProcessor(config)
        visualizer = Visualizer(config)
        
        # Validate data
        try:
            data_loader.validate_data(df)
            
            with st.spinner('Processing texts...'):
                # Process data
                deduplicated_df, dedup_info, similarity_matrix, embeddings, labels = text_processor.process(df)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Texts", len(df))
                with col2:
                    st.metric("Unique Texts", len(deduplicated_df))
                with col3:
                    st.metric("Duplicates Removed", dedup_info['duplicates_removed'])
                
                # Text Length Distribution
                st.header("Text Length Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    fig = visualizer.plot_text_length_distribution(df, "Original Texts")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = visualizer.plot_text_length_distribution(deduplicated_df, "Deduplicated Texts")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Similarity Analysis
                st.header("Similarity Analysis")
                fig = visualizer.plot_similarity_heatmap(
                    similarity_matrix,
                    df[config['data']['text_column']].tolist()
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster Visualization
                st.header("Cluster Visualization")
                fig = visualizer.plot_cluster_visualization(
                    embeddings,
                    labels,
                    df[config['data']['text_column']].tolist()
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Deduplication Summary
                st.header("Deduplication Summary")
                fig = visualizer.plot_deduplication_summary(dedup_info)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sample Duplicates
                st.header("Sample Duplicate Groups")
                for original, duplicates in list(dedup_info['duplicate_mapping'].items())[:5]:
                    with st.expander(f"Original: {original}"):
                        st.write("Duplicates:")
                        for dup in duplicates:
                            st.write(f"- {dup}")
                
                # Download deduplicated data
                st.download_button(
                    "Download Deduplicated Data",
                    deduplicated_df.to_csv(index=False),
                    "deduplicated_data.csv",
                    "text/csv",
                    key='download-csv'
                )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis")
        
        # Show sample data format
        st.header("Sample Data Format")
        sample_data = pd.DataFrame({
            'texts': [
                "How can I help you?",
                "I've been charged twice",
                "Thank you for calling"
            ],
            'labels': [
                "non-complaint",
                "complaint",
                "non-complaint"
            ]
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main() 