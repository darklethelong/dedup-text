import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st
from sklearn.manifold import TSNE
import logging

# Define a consistent color scheme and styling
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'complaint': '#d62728',
    'non-complaint': '#2ca02c',
    'background': '#f8f9fa',
    'grid': '#e9ecef'
}

PLOT_LAYOUT = {
    'font_family': 'Arial',
    'title_font_size': 24,
    'paper_bgcolor': COLORS['background'],
    'plot_bgcolor': COLORS['background'],
    'showlegend': True,
    'legend_bgcolor': 'rgba(255, 255, 255, 0.8)',
    'margin': dict(t=100, l=40, r=40, b=40)
}

class Visualizer:
    def __init__(self, config: dict):
        """Initialize Visualizer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def plot_text_length_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Text Length Distribution"
    ) -> go.Figure:
        """Plot enhanced distribution of text lengths."""
        try:
            text_lengths = df[self.config['data']['text_column']].str.len()
            labels = df[self.config['data']['label_column']]
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add histogram for all texts
            for label in labels.unique():
                mask = labels == label
                fig.add_trace(go.Histogram(
                    x=text_lengths[mask],
                    name=f'{label.title()}',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=COLORS['complaint'] if label == 'complaint' else COLORS['non-complaint']
                ))
            
            # Add smoothed line using moving average instead of KDE
            sorted_lengths = np.sort(text_lengths)
            window = max(len(text_lengths) // 20, 5)  # Dynamic window size
            moving_avg = pd.Series(sorted_lengths).rolling(window=window, center=True).mean()
            
            fig.add_trace(go.Scatter(
                x=sorted_lengths,
                y=moving_avg,
                name='Trend',
                line=dict(color='black', width=2),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=24)
                ),
                xaxis_title="Text Length (characters)",
                yaxis_title="Count",
                yaxis2=dict(
                    title="Moving Average",
                    overlaying="y",
                    side="right"
                ),
                **PLOT_LAYOUT,
                barmode='overlay'
            )
            
            # Update axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting text length distribution: {str(e)}")
            raise

    def plot_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        texts: List[str],
        title: str = "Text Similarity Heatmap"
    ) -> go.Figure:
        """Plot enhanced similarity matrix heatmap."""
        try:
            # Sample if too many texts
            max_samples = self.config['visualization']['similarity_heatmap_max_samples']
            if len(texts) > max_samples:
                indices = np.random.choice(len(texts), max_samples, replace=False)
                similarity_matrix = similarity_matrix[indices][:, indices]
                texts = [texts[i] for i in indices]

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=texts,
                y=texts,
                colorscale='Viridis',
                text=np.around(similarity_matrix, decimals=2),
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='Text 1: %{x}<br>Text 2: %{y}<br>Similarity: %{z:.2f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=24)
                ),
                **PLOT_LAYOUT,
                width=900,
                height=800
            )
            
            # Update axes
            fig.update_xaxes(showticklabels=False, showgrid=False)
            fig.update_yaxes(showticklabels=False, showgrid=False)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting similarity heatmap: {str(e)}")
            raise

    def plot_cluster_visualization(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        title: str = "Text Clusters Visualization"
    ) -> go.Figure:
        """Plot enhanced t-SNE visualization of text clusters."""
        try:
            # Sample if too many points
            max_samples = self.config['visualization']['cluster_viz_max_samples']
            if len(texts) > max_samples:
                indices = np.random.choice(len(texts), max_samples, replace=False)
                embeddings = embeddings[indices]
                labels = labels[indices]
                texts = [texts[i] for i in indices]

            # Compute t-SNE with better parameters
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(texts)-1),
                n_iter=1000
            )
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels],
                'text': texts
            })
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add traces for each cluster
            for cluster in sorted(df_plot['cluster'].unique()):
                mask = df_plot['cluster'] == cluster
                cluster_data = df_plot[mask]
                
                fig.add_trace(go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    name=cluster,
                    marker=dict(
                        size=10,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=cluster_data['text'],
                    hovertemplate='%{text}<br>Cluster: ' + cluster + '<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=24)
                ),
                xaxis_title="t-SNE Dimension 1",
                yaxis_title="t-SNE Dimension 2",
                **PLOT_LAYOUT,
                width=900,
                height=700
            )
            
            # Update axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'], zeroline=False)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'], zeroline=False)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting cluster visualization: {str(e)}")
            raise

    def plot_deduplication_summary(
        self,
        dedup_info: Dict,
        title: str = "Deduplication Summary"
    ) -> go.Figure:
        """Plot enhanced summary statistics of deduplication process."""
        try:
            # Prepare summary data
            summary_data = [
                {
                    'Metric': 'Total Clusters',
                    'Value': dedup_info['total_clusters'],
                    'Description': 'Number of unique text groups'
                },
                {
                    'Metric': 'Duplicates Removed',
                    'Value': dedup_info['duplicates_removed'],
                    'Description': 'Number of similar texts removed'
                },
                {
                    'Metric': 'Average Cluster Size',
                    'Value': np.mean(list(dedup_info['cluster_sizes'].values())),
                    'Description': 'Average number of texts per cluster'
                }
            ]
            
            # Create figure with subplots
            fig = go.Figure()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=[d['Metric'] for d in summary_data],
                y=[d['Value'] for d in summary_data],
                text=[f"{d['Value']:.1f}" for d in summary_data],
                textposition='auto',
                marker_color=[COLORS['primary'], COLORS['secondary'], COLORS['complaint']],
                hovertemplate='%{x}<br>Value: %{y:.1f}<br>%{customdata}<extra></extra>',
                customdata=[d['Description'] for d in summary_data]
            ))
            
            # Add cluster size distribution
            cluster_sizes = list(dedup_info['cluster_sizes'].values())
            if cluster_sizes:
                fig.add_trace(go.Violin(
                    y=cluster_sizes,
                    name='Cluster Size Distribution',
                    side='positive',
                    line_color=COLORS['secondary'],
                    showlegend=False
                ))
            
            # Update layout
            layout_settings = {
                'title': dict(
                    text=title,
                    x=0.5,
                    font=dict(size=24)
                ),
                'yaxis_title': "Count",
                'showlegend': False
            }
            layout_settings.update({k: v for k, v in PLOT_LAYOUT.items() if k != 'showlegend'})
            fig.update_layout(**layout_settings)
            
            # Update axes
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting deduplication summary: {str(e)}")
            raise

    def create_streamlit_dashboard(
        self,
        original_df: pd.DataFrame,
        deduplicated_df: pd.DataFrame,
        similarity_matrix: np.ndarray,
        embeddings: np.ndarray,
        labels: np.ndarray,
        dedup_info: Dict
    ) -> None:
        """Create enhanced interactive Streamlit dashboard."""
        try:
            st.title("Text Deduplication Analysis Dashboard")
            
            # Add description
            st.markdown("""
            This dashboard provides an interactive visualization of the text deduplication process.
            The analysis includes text length distributions, similarity patterns, and cluster information.
            """)
            
            # Dataset Overview with enhanced metrics
            st.header("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Original Texts",
                    len(original_df),
                    delta=None,
                    help="Total number of texts before deduplication"
                )
            with col2:
                st.metric(
                    "Unique Texts",
                    len(deduplicated_df),
                    delta=f"-{dedup_info['duplicates_removed']}",
                    delta_color="inverse",
                    help="Number of texts after removing duplicates"
                )
            with col3:
                duplicate_ratio = (dedup_info['duplicates_removed'] / len(original_df)) * 100
                st.metric(
                    "Duplicate Ratio",
                    f"{duplicate_ratio:.1f}%",
                    help="Percentage of texts identified as duplicates"
                )
            with col4:
                avg_cluster_size = np.mean(list(dedup_info['cluster_sizes'].values()))
                st.metric(
                    "Avg. Cluster Size",
                    f"{avg_cluster_size:.1f}",
                    help="Average number of similar texts per cluster"
                )
                
            # Text Length Distribution
            st.header("📏 Text Length Distribution")
            fig = self.plot_text_length_distribution(original_df, "Text Length Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
                
            # Similarity Analysis
            st.header("🔍 Similarity Analysis")
            fig = self.plot_similarity_heatmap(
                similarity_matrix,
                original_df[self.config['data']['text_column']].tolist()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster Visualization
            st.header("🎯 Cluster Visualization")
            fig = self.plot_cluster_visualization(
                embeddings,
                labels,
                original_df[self.config['data']['text_column']].tolist()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Deduplication Summary
            st.header("📈 Deduplication Summary")
            fig = self.plot_deduplication_summary(dedup_info)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample Duplicates with enhanced UI
            st.header("🔄 Sample Duplicate Groups")
            for original, duplicates in list(dedup_info['duplicate_mapping'].items())[:5]:
                with st.expander(f"📝 Original: {original}"):
                    st.markdown("**Similar texts found:**")
                    for i, dup in enumerate(duplicates, 1):
                        st.markdown(f"{i}. {dup}")
                    st.markdown(f"*Similarity score: {self.config['deduplication']['initial_threshold']:.2f}*")
            
            # Add download section with description
            st.header("💾 Download Results")
            st.markdown("""
            Download the deduplicated dataset in CSV format.
            The file contains unique representative texts and their corresponding labels.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Deduplicated Data",
                    deduplicated_df.to_csv(index=False),
                    "deduplicated_data.csv",
                    "text/csv",
                    key='download-csv'
                )
            with col2:
                st.markdown(f"*File contains {len(deduplicated_df)} unique texts*")
                
        except Exception as e:
            self.logger.error(f"Error creating Streamlit dashboard: {str(e)}")
            raise 