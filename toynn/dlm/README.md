# Toy Diffusion Language Model (DLM) Implementation

A minimal implementation of a Diffusion Language Model with masked token diffusion for text generation and refinement.

## Features

- **Masked Token Diffusion**: Progressive masking/unmasking of discrete tokens
- **Transformer-based Denoising**: Full transformer network for token prediction
- **Confidence-based Unmasking**: Strategic token revelation based on prediction confidence
- **PDF Data Processing**: Extract and process text from PDF documents
- **HRM Integration Ready**: Can be extended with H/L module hierarchy

## Project Structure

```
dlm/
├── model/              # Model architecture
│   ├── __init__.py
│   ├── diffusion.py    # Main DLM model
│   ├── transformer.py  # Transformer components
│   └── utils.py        # Helper functions
├── data/               # Data processing
│   ├── __init__.py
│   ├── dataset.py      # Dataset classes
│   └── pdf_processor.py # PDF text extraction
├── training/           # Training scripts
│   ├── __init__.py
│   ├── train.py        # Training loop
│   └── config.py       # Training configuration
├── demo.ipynb          # Interactive demo
└── requirements.txt    # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Process PDF Data**:
```python
from data.pdf_processor import PDFProcessor
processor = PDFProcessor()
texts = processor.process_folder("path/to/pdfs")
```

2. **Train Model**:
```bash
python training/train.py --data_path data/processed --epochs 100
```

3. **Generate Text**:
```python
from model.diffusion import DiffusionLM
model = DiffusionLM.load("checkpoints/model.pt")
text = model.generate("Start of text", max_length=100)
```

## Architecture

The model uses masked token diffusion with:
- **Forward Process**: Progressively mask tokens with increasing probability
- **Reverse Process**: Iteratively predict and unmask tokens using transformer
- **Timestep Conditioning**: Sinusoidal embeddings guide denoising strength

## Key Components

### Masking Strategy
- Random masking during training
- Confidence-based unmasking during generation
- Progressive schedule: fewer → more masks

### Token Prediction
- Bidirectional transformer for parallel prediction
- Timestep-aware through adaptive layer normalization
- Confidence scoring for strategic unmasking

## References

- Diffusion Language Models
- Masked Diffusion for Text Generation
- Score-based Discrete Diffusion
