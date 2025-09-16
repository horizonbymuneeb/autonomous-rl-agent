"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Update evaluation protocol for robustness checks [2025-06-11T14:39:41]

# Implement DQN with experience replay buffer [2025-06-16T20:01:06]

# Add transfer learning between related tasks [2025-06-20T09:49:34]

# WIP: tuning hyperparameters for sparse rewards [2025-06-20T14:16:45]

# Implement hierarchical RL for long horizons [2025-06-20T13:16:39]

# Implement SAC for off-policy learning [2025-06-30T11:41:52]

# WIP: tuning hyperparameters for sparse rewards [2025-07-02T12:54:09]

# WIP: debugging reward shaping for stability [2025-07-07T14:03:22]

# Implement model-based RL with learned dynamics [2025-07-08T14:53:02]

# Implement curriculum learning for harder tasks [2025-07-09T20:00:45]

# Implement hindsight experience replay for sparse [2025-07-09T17:29:18]

# Add custom Gym environment for navigation [2025-07-10T10:58:41]

# Implement hindsight experience replay for sparse [2025-07-11T20:18:09]

# WIP: debugging reward shaping for stability [2025-07-13T10:30:50]

# Update evaluation protocol for robustness checks [2025-07-17T11:00:44]

# Implement DQN with experience replay buffer [2025-07-18T20:44:18]

# Implement hindsight experience replay for sparse [2025-07-21T17:20:20]

# WIP: benchmarking on MuJoCo control tasks [2025-07-24T20:10:41]

# Implement SAC for off-policy learning [2025-07-28T18:06:55]

# Update evaluation protocol for robustness checks [2025-08-01T16:17:13]

# Update README with usage and setup instructions [2025-08-03T17:59:27]

# Implement DQN with experience replay buffer [2025-08-05T13:04:07]

# Add transfer learning between related tasks [2025-08-08T18:38:21]

# WIP: benchmarking on MuJoCo control tasks [2025-08-15T18:35:56]

# Implement hierarchical RL for long horizons [2025-08-18T13:53:21]

# Implement hierarchical RL for long horizons [2025-08-22T13:28:23]

# Add safety layer for real-world deployment [2025-08-26T20:06:59]

# Update README with usage and setup instructions [2025-08-26T12:43:38]

# Add safety layer for real-world deployment [2025-08-28T16:37:20]

# Update PPO for continuous action control [2025-09-03T17:09:36]

# Implement SAC for off-policy learning [2025-09-03T10:07:41]

# Implement DQN with experience replay buffer [2025-09-05T13:51:04]

# Fix state representation bug in environment [2025-09-08T17:56:49]

# Implement hindsight experience replay for sparse [2025-09-10T10:50:57]

# WIP: benchmarking on MuJoCo control tasks [2025-09-10T17:04:16]

# Implement DQN with experience replay buffer [2025-09-12T12:15:02]

# Update evaluation protocol for robustness checks [2025-09-16T14:23:11]
