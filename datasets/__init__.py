"""
KitAI Datasets Package.

Provides dataset loaders for various data formats:
- Plain text (.txt)
- JSONL (.jsonl)
- Parquet (.parquet)
- CSV (.csv)
- Alpaca format
- ShareGPT format
- OpenAI messages format
- Sequence packing utility
"""

from .base_dataset import BaseDataset
from .text_dataset import TextFileDataset
from .jsonl_dataset import JSONLDataset
from .parquet_dataset import ParquetDataset
from .csv_dataset import CSVDataset
from .alpaca_dataset import AlpacaDataset
from .sharegpt_dataset import ShareGPTDataset
from .openai_dataset import OpenAIDataset
from .packer import SequencePacker

__all__ = [
    "BaseDataset",
    "TextFileDataset",
    "JSONLDataset",
    "ParquetDataset",
    "CSVDataset",
    "AlpacaDataset",
    "ShareGPTDataset",
    "OpenAIDataset",
    "SequencePacker",
]
