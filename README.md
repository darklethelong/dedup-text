# Text Deduplication for Call Center Utterances

A professional tool for detecting and removing semantically similar texts from call center data, with advanced visualization capabilities.

## Features

- **Semantic Similarity Detection**: Uses state-of-the-art sentence transformers to identify similar texts
- **Adaptive Thresholding**: Automatically adjusts similarity thresholds based on text length
- **Advanced Clustering**: Implements both agglomerative and DBSCAN clustering
- **Interactive Visualization**: Comprehensive Streamlit dashboard for data analysis
- **Robust Error Handling**: Extensive validation and error tracking
- **Progress Tracking**: Visual feedback during processing
- **Configurable**: Easily customizable through YAML configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dedup-text.git
cd dedup-text
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python src/main.py --input data/raw/input.csv --output data/processed/output.csv
```

### With Visualization

```bash
python src/main.py --input data/raw/input.csv --visualize
```

### Configuration

Modify `config.yaml` to customize:
- Similarity thresholds
- Clustering parameters
- Visualization settings
- Model selection
- Input/output paths

## Input Format

The input CSV file should contain two columns:
- `texts`: The text utterances
- `labels`: Corresponding labels

Example:
```csv
texts,labels
"how can I help you?",greeting
"my name is John",introduction
```

## Output

The tool provides:
1. Deduplicated CSV file
2. Interactive visualization dashboard
3. Processing logs
4. Summary statistics

## Visualization Dashboard

The dashboard includes:
- Text length distribution
- Similarity heatmap
- Cluster visualization
- Deduplication summary
- Sample duplicate groups

## Advanced Features

### Adaptive Thresholding

The tool automatically adjusts similarity thresholds based on:
- Text length
- Common phrases
- Cluster density

### Clustering Methods

Two clustering approaches available:
1. **Agglomerative Clustering**: For hierarchical grouping
2. **DBSCAN**: For density-based clustering

## Performance

- Efficiently handles large datasets through batch processing
- GPU acceleration when available
- Memory-efficient similarity computation

## Error Handling

- Comprehensive input validation
- Detailed error logging
- Graceful failure recovery

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

Your Name - your.email@example.com 