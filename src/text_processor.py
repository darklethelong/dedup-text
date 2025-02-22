import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import os
from typing import List, Tuple, Dict

# Import ML libraries after setting environment variables
os.environ["PYTORCH_JIT"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

class TextProcessor:
    def __init__(self, config: dict):
        """Initialize TextProcessor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = self._get_device()
        self.model = self._load_model()
        
    def _get_device(self) -> str:
        """Determine the device to use for computations."""
        device = "cpu"
        if self.config['model']['device'] == "auto":
            try:
                # Import torch only when needed and in a controlled way
                import torch.cuda
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass
        return device

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            model = SentenceTransformer(
                self.config['model']['name'],
                device=self.device
            )
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        try:
            self.logger.info("Computing embeddings...")
            embeddings = []
            
            # Process in batches to manage memory
            batch_size = self.config['model']['batch_size']
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    device=self.device
                )
                embeddings.append(batch_embeddings)
                
            return np.vstack(embeddings)
            
        except Exception as e:
            self.logger.error(f"Error computing embeddings: {str(e)}")
            raise

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between embeddings."""
        try:
            self.logger.info("Computing similarity matrix...")
            return cosine_similarity(embeddings)
        except Exception as e:
            self.logger.error(f"Error computing similarity matrix: {str(e)}")
            raise

    def _get_adaptive_threshold(self, text_length: int) -> float:
        """Calculate adaptive threshold based on text length."""
        base_threshold = self.config['deduplication']['initial_threshold']
        min_threshold = self.config['deduplication']['min_threshold']
        max_threshold = self.config['deduplication']['max_threshold']
        
        # Shorter texts need higher threshold to avoid false positives
        if text_length < 10:
            threshold = base_threshold + 0.05
        elif text_length < 20:
            threshold = base_threshold
        else:
            threshold = base_threshold - 0.05
            
        return np.clip(threshold, min_threshold, max_threshold)

    def cluster_texts(self, similarity_matrix: np.ndarray, texts: List[str]) -> np.ndarray:
        """Cluster texts based on similarity matrix."""
        try:
            self.logger.info("Clustering similar texts...")
            
            if self.config['clustering']['method'] == 'agglomerative':
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - self.config['deduplication']['initial_threshold'],
                    metric='precomputed',
                    linkage='complete'
                )
                distances = 1 - similarity_matrix
                labels = clustering.fit_predict(distances)
                
            elif self.config['clustering']['method'] == 'dbscan':
                clustering = DBSCAN(
                    eps=self.config['deduplication']['initial_threshold'],
                    min_samples=self.config['clustering']['min_cluster_size'],
                    metric='precomputed'
                )
                labels = clustering.fit_predict(1 - similarity_matrix)
            
            else:
                raise ValueError(f"Unknown clustering method: {self.config['clustering']['method']}")
                
            return labels
            
        except Exception as e:
            self.logger.error(f"Error during clustering: {str(e)}")
            raise

    def find_cluster_representatives(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict]:
        """Find representative texts for each cluster."""
        try:
            self.logger.info("Finding cluster representatives...")
            
            # Initialize tracking dictionary
            dedup_info = {
                'total_clusters': len(set(labels)),
                'duplicates_removed': 0,
                'cluster_sizes': {},
                'duplicate_mapping': {}
            }
            
            # Handle noise points (label -1) first
            mask = labels != -1
            unique_texts = df[~mask].copy()
            
            # Process each cluster
            for cluster_id in set(labels[mask]):
                cluster_mask = labels == cluster_id
                cluster_size = np.sum(cluster_mask)
                dedup_info['cluster_sizes'][cluster_id] = cluster_size
                
                if cluster_size >= self.config['clustering']['min_cluster_size']:
                    # Find the text closest to cluster centroid
                    cluster_embeddings = embeddings[cluster_mask]
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                    representative_idx = np.argmin(distances)
                    
                    # Get cluster texts and their indices
                    cluster_texts = df[cluster_mask]
                    representative = cluster_texts.iloc[representative_idx]
                    duplicates = cluster_texts.drop(representative.name)
                    
                    # Update tracking info
                    dedup_info['duplicates_removed'] += len(duplicates)
                    dedup_info['duplicate_mapping'][representative[self.config['data']['text_column']]] = \
                        duplicates[self.config['data']['text_column']].tolist()
                    
                    # Add representative to unique texts
                    unique_texts = pd.concat([unique_texts, representative.to_frame().T])
                else:
                    # Keep all texts from small clusters
                    unique_texts = pd.concat([unique_texts, df[cluster_mask]])
            
            self.logger.info(f"Removed {dedup_info['duplicates_removed']} duplicate texts")
            return unique_texts.reset_index(drop=True), dedup_info
            
        except Exception as e:
            self.logger.error(f"Error finding cluster representatives: {str(e)}")
            raise

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Main processing pipeline."""
        try:
            texts = df[self.config['data']['text_column']].tolist()
            
            # Compute embeddings
            embeddings = self.compute_embeddings(texts)
            
            # Compute similarity matrix
            similarity_matrix = self.compute_similarity_matrix(embeddings)
            
            # Cluster similar texts
            labels = self.cluster_texts(similarity_matrix, texts)
            
            # Find representatives and deduplicate
            deduplicated_df, dedup_info = self.find_cluster_representatives(
                df, embeddings, labels
            )
            
            return deduplicated_df, dedup_info, similarity_matrix, embeddings, labels
            
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            raise 