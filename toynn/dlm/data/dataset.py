"""Dataset classes for DLM training."""

import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import random
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Basic text dataset for language modeling."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        cache_data: bool = True
    ):
        """Initialize text dataset.
        
        Args:
            data_path: Path to data file or directory
            tokenizer: Tokenizer to use (will create if None)
            max_length: Maximum sequence length
            cache_data: Whether to cache processed data
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.cache_data = cache_data
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        # Add mask token if not present
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        
        # Load data
        self.texts = self._load_data()
        
        # Cache tokenized data
        self.cached_data = {}
        if cache_data:
            print("Caching tokenized data...")
            for idx in range(len(self.texts)):
                self.cached_data[idx] = self._tokenize(self.texts[idx])
    
    def _load_data(self) -> List[str]:
        """Load text data from file or directory.
        
        Returns:
            List of text strings
        """
        texts = []
        
        if self.data_path.is_file():
            # Load from single file
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Handle list of documents
                        for doc in data:
                            if isinstance(doc, dict) and 'chunks' in doc:
                                texts.extend(doc['chunks'])
                            elif isinstance(doc, dict) and 'text' in doc:
                                texts.append(doc['text'])
                            elif isinstance(doc, str):
                                texts.append(doc)
                    elif isinstance(data, dict) and 'texts' in data:
                        texts = data['texts']
            
            elif self.data_path.suffix == '.txt':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    # Split by double newline
                    texts = f.read().split('\n\n')
        
        elif self.data_path.is_dir():
            # Load from directory
            for file_path in self.data_path.glob('**/*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            
            for file_path in self.data_path.glob('**/*.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chunks' in data:
                        texts.extend(data['chunks'])
                    elif 'text' in data:
                        texts.append(data['text'])
        
        # Filter by length
        texts = [t for t in texts if len(t.strip()) > 10]
        
        print(f"Loaded {len(texts)} text samples")
        return texts
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with token IDs and attention mask
        """
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_data and idx in self.cached_data:
            return self.cached_data[idx]
        else:
            return self._tokenize(self.texts[idx])


class DiffusionDataset(Dataset):
    """Dataset for diffusion language model training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        mask_token_id: Optional[int] = None,
        cache_data: bool = True
    ):
        """Initialize diffusion dataset.
        
        Args:
            data_path: Path to data
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            mask_token_id: ID for mask token
            cache_data: Whether to cache data
        """
        # Initialize base dataset
        self.base_dataset = TextDataset(
            data_path, tokenizer, max_length, cache_data
        )
        
        self.tokenizer = self.base_dataset.tokenizer
        self.max_length = max_length
        
        # Set mask token ID
        if mask_token_id is None:
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token_id = mask_token_id
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item for training.
        
        Returns:
            Dictionary with:
                - input_ids: Original token IDs
                - attention_mask: Attention mask
        """
        item = self.base_dataset[idx]
        
        # Return clean tokens (masking done in model during training)
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask']
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def create_dataloader(
    data_path: str,
    batch_size: int = 8,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create dataloader for training.
    
    Args:
        data_path: Path to data
        batch_size: Batch size
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader
    """
    dataset = DiffusionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader
