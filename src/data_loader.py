import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import yaml

class DataLoader:
    def __init__(self, config: Union[str, Dict] = "config.yaml"):
        """Initialize DataLoader with configuration.
        
        Args:
            config: Either a path to config file (str) or config dictionary
        """
        self.config = self._load_config(config)
        self._setup_logging()
        
    def _load_config(self, config: Union[str, Dict]) -> dict:
        """Load configuration from YAML file or use provided dict."""
        try:
            if isinstance(config, dict):
                return config
            
            with open(config, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger = logging.getLogger(__name__)
            self.logger.error(f"Error loading config: {str(e)}")
            raise ValueError(f"Error loading config: {str(e)}")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler() if self.config['logging']['console'] else None
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the input dataframe structure and content."""
        try:
            # Check required columns
            required_cols = [
                self.config['data']['text_column'],
                self.config['data']['label_column']
            ]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")

            # Check for empty dataframe
            if df.empty:
                raise ValueError("Empty dataframe provided")

            # Check for null values
            null_counts = df[required_cols].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Found null values: {null_counts}")

            # Validate text length
            text_lengths = df[self.config['data']['text_column']].str.len()
            invalid_lengths = (
                (text_lengths < self.config['deduplication']['min_text_length']) |
                (text_lengths > self.config['deduplication']['max_text_length'])
            )
            if invalid_lengths.any():
                self.logger.warning(
                    f"Found {invalid_lengths.sum()} texts with invalid lengths"
                )

            return True
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, Optional[str]]:
        """Load and validate the input data."""
        try:
            input_file = Path(self.config['data']['input_file'])
            
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Determine file type and read accordingly
            if input_file.suffix == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(input_file)
            else:
                raise ValueError(f"Unsupported file format: {input_file.suffix}")

            # Validate the loaded data
            self.validate_data(df)
            
            # Basic cleaning
            text_col = self.config['data']['text_column']
            df[text_col] = df[text_col].astype(str).str.strip()
            
            self.logger.info(f"Successfully loaded {len(df)} records from {input_file}")
            return df, None

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            self.logger.error(error_msg)
            return pd.DataFrame(), error_msg

    def save_data(self, df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """Save the processed dataframe."""
        try:
            output_file = Path(output_path or self.config['data']['output_file'])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_file, index=False)
            self.logger.info(f"Successfully saved {len(df)} records to {output_file}")
            
        except Exception as e:
            error_msg = f"Error saving data: {str(e)}"
            self.logger.error(error_msg)
            raise 