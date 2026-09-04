"""
CSV Dataset.

Loads CSV files for training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CSVDataset(BaseDataset):
    """
    Dataset for CSV files.

    Args:
        file_paths: Path(s) to CSV file(s).
        block_size: Maximum sequence length.
        tokenizer_encode_fn: Optional encoding function.
        text_field: Column name containing the text (default: "text").

    Examples:
        >>> ds = CSVDataset("data.csv", block_size=128)
        >>> inputs, labels = ds[0]
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        block_size: int = 2048,
        tokenizer_encode_fn: Optional[Callable[[str], List[int]]] = None,
        text_field: str = "text",
    ) -> None:
        super().__init__(block_size, tokenizer_encode_fn)
        self.text_field = text_field

        if isinstance(file_paths, (str, Path)):
            path = Path(file_paths)
            if path.is_dir():
                self.file_paths = sorted(path.glob("*.csv"))
            else:
                self.file_paths = [path]
        else:
            self.file_paths = [Path(p) for p in file_paths]

        if not self.file_paths:
            raise ValueError(f"No .csv files found: {file_paths}")

        self.load_data()

    def load_data(self) -> None:
        """Read CSV files and create training samples."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV support. Install with: pip install pandas")

        all_tokens = []
        for fp in self.file_paths:
            if fp.exists():
                try:
                    df = pd.read_csv(fp)
                    if self.text_field not in df.columns:
                        logger.warning(f"Column '{self.text_field}' not found in {fp}. Columns: {list(df.columns)}")
                        continue
                    texts = df[self.text_field].dropna().tolist()
                    for text in texts:
                        all_tokens.extend(self._encode(str(text)))
                        all_tokens.append(0)
                    logger.info(f"Loaded {fp.name}: {len(texts)} rows")
                except Exception as e:
                    logger.warning(f"Failed to read {fp}: {e}")

        for i in range(0, len(all_tokens) - self.block_size, self.block_size // 2):
            chunk = all_tokens[i : i + self.block_size + 1]
            if len(chunk) >= self.block_size + 1:
                self.samples.append(self._create_sample(chunk))
