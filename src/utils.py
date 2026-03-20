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

# Implement curriculum learning for harder tasks [2025-09-18T15:57:51]

# Add safety layer for real-world deployment [2025-09-19T16:45:06]

# Add safety layer for real-world deployment [2025-09-22T11:28:37]

# Add logging for training metrics to tensorboard [2025-09-23T11:18:04]

# Update evaluation protocol for robustness checks [2025-09-23T17:05:14]

# Update agent for multi-agent scenario support [2025-09-25T14:57:55]

# Add custom Gym environment for navigation [2025-09-29T14:21:04]

# Update agent for multi-agent scenario support [2025-10-01T14:52:40]

# Implement curriculum learning for harder tasks [2025-10-06T20:00:34]

# Add logging for training metrics to tensorboard [2025-10-07T11:30:30]

# Implement hindsight experience replay for sparse [2025-10-09T10:54:10]

# Fix state representation bug in environment [2025-10-13T18:50:31]

# Update PPO for continuous action control [2025-10-16T14:14:18]

# Add safety layer for real-world deployment [2025-10-21T11:40:10]

# Add safety layer for real-world deployment [2025-10-22T12:21:12]

# Add PyBullet physics simulation backend [2025-10-24T17:44:30]

# WIP: debugging reward shaping for stability [2025-10-26T09:18:10]

# WIP: debugging reward shaping for stability [2025-10-31T17:46:12]

# Add logging for training metrics to tensorboard [2025-11-11T13:16:33]

# Implement curriculum learning for harder tasks [2025-11-12T10:11:08]

# WIP: debugging reward shaping for stability [2025-11-13T10:14:49]

# Fix state representation bug in environment [2025-11-19T16:27:25]

# Add PyBullet physics simulation backend [2025-11-22T15:17:40]

# Update README with usage and setup instructions [2025-11-22T14:57:51]

# Implement SAC for off-policy learning [2025-11-27T20:06:15]

# WIP: benchmarking on MuJoCo control tasks [2025-11-30T17:44:11]

# Update README with usage and setup instructions [2025-11-30T20:51:05]

# Update PPO for continuous action control [2025-12-14T09:33:31]

# Update README with usage and setup instructions [2025-12-15T18:01:55]

# Add PyBullet physics simulation backend [2025-12-20T09:42:37]

# Add custom Gym environment for navigation [2025-12-23T15:08:50]

# Update agent for multi-agent scenario support [2025-12-29T09:00:51]

# Implement SAC for off-policy learning [2025-12-30T16:56:51]

# Add transfer learning between related tasks [2026-01-05T19:35:28]

# Update agent for multi-agent scenario support [2026-01-08T10:49:58]

# Implement hierarchical RL for long horizons [2026-01-13T17:39:05]

# Implement hindsight experience replay for sparse [2026-01-16T19:57:58]

# Implement hierarchical RL for long horizons [2026-01-19T18:16:11]

# WIP: debugging reward shaping for stability [2026-01-20T10:54:29]

# Implement DQN with experience replay buffer [2026-01-22T16:49:35]

# WIP: debugging reward shaping for stability [2026-01-29T09:00:02]

# Update agent for multi-agent scenario support [2026-01-30T17:17:16]

# Add safety layer for real-world deployment [2026-02-01T14:24:54]

# Add safety layer for real-world deployment [2026-02-04T11:14:27]

# Fix observation normalization statistics [2026-02-09T18:10:14]

# Fix observation normalization statistics [2026-02-10T10:15:19]

# Implement hindsight experience replay for sparse [2026-02-12T19:25:36]

# Update agent for multi-agent scenario support [2026-02-19T13:49:38]

# WIP: tuning hyperparameters for sparse rewards [2026-02-21T09:53:40]

# Add transfer learning between related tasks [2026-02-23T13:07:21]

# Implement model-based RL with learned dynamics [2026-02-27T13:51:07]

# WIP: debugging reward shaping for stability [2026-02-27T13:58:45]

# Add custom Gym environment for navigation [2026-03-09T09:39:33]

# Add safety layer for real-world deployment [2026-03-15T12:18:35]

# Update PPO for continuous action control [2026-03-18T20:13:42]

# Implement hierarchical RL for long horizons [2026-03-18T20:16:44]

# Implement SAC for off-policy learning [2026-03-19T09:43:38]

# Add PyBullet physics simulation backend [2026-03-19T16:06:46]

# WIP: debugging reward shaping for stability [2026-03-20T15:08:59]

# WIP: debugging reward shaping for stability [2026-03-20T19:06:22]
