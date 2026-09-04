"""
File I/O utilities for KitAI.

Provides helper functions for reading/writing configuration files,
checkpoints, and other data formats used throughout the project.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded YAML config from {path}")
    return config if config is not None else {}


def write_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Write a dictionary to a YAML file.

    Args:
        data: Data to write.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Wrote YAML config to {path}")


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def write_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        data: Data to write.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Wrote JSON to {path}")


def read_jsonl(path: Union[str, Path]) -> list:
    """
    Read a JSONL (JSON Lines) file.

    Args:
        path: Path to the JSONL file.

    Returns:
        list: List of parsed JSON objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def write_jsonl(data: list, path: Union[str, Path]) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    logger.debug(f"Wrote {len(data)} records to {path}")


def read_text(path: Union[str, Path]) -> str:
    """
    Read a plain text file.

    Args:
        path: Path to the text file.

    Returns:
        str: File contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    with open(path, "r") as f:
        return f.read()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path: The resolved directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the project root if not absolute.

    Args:
        path: Path to resolve.

    Returns:
        Path: Resolved absolute path.
    """
    path = Path(path)
    if not path.is_absolute():
        # Try to find project root (where configs/ lives)
        project_root = Path.cwd()
        path = project_root / path
    return path.resolve()
