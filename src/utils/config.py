"""
Configuration loader and system utilities.
"""

from pathlib import Path
from typing import Any, Dict
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file safely.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dict containing configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        # Fallback to project root relative to this module
        project_root = Path(__file__).resolve().parent.parent.parent
        fallback = project_root / config_path
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Configuration file not found at '{path.resolve()}' or '{fallback.resolve()}'")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device(preferred_device: str = "auto") -> torch.device:
    """
    Determine the optimal compute device (CUDA GPU or CPU).

    Args:
        preferred_device: 'auto', 'cuda', 'cpu', or device index like '0'.

    Returns:
        torch.device instance.
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[System] CUDA detected: using GPU ({device_name})")
            return torch.device("cuda:0")
        else:
            print("[System] CUDA not available: falling back to CPU")
            return torch.device("cpu")

    elif str(preferred_device).isdigit() or "cuda" in str(preferred_device):
        if torch.cuda.is_available():
            dev_str = f"cuda:{preferred_device}" if str(preferred_device).isdigit() else str(preferred_device)
            return torch.device(dev_str)
        else:
            print(f"[System Warning] Requested {preferred_device} but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")
