"""
Sequence Dataset for Transformer Models

Converts tabular time series data into windowed sequences
suitable for Transformer-based models (PyTorch Dataset).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class TimeSeriesSequenceDataset(Dataset):
    """
    Sliding-window dataset for time series Transformer models.

    Converts a DataFrame with features + target into (X_window, y_target) pairs
    where X_window has shape (seq_len, n_features) and y_target is the next value.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 60,
        horizon: int = 1,
        stride: int = 1,
    ):
        """
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            seq_len: Length of input sequence window
            horizon: Number of steps to predict ahead
            stride: Step size between consecutive windows
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.horizon = horizon
        self.stride = stride

        # Calculate valid indices
        self.indices = list(
            range(0, len(features) - seq_len - horizon + 1, stride)
        )

        if len(self.indices) == 0:
            raise ValueError(
                f"Not enough data for seq_len={seq_len}, horizon={horizon}. "
                f"Got {len(features)} samples, need at least {seq_len + horizon}."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len

        X = self.features[start:end]  # (seq_len, n_features)
        y = self.targets[end:end + self.horizon]  # (horizon,)

        return X, y


def create_sequence_dataloaders(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "close",
    seq_len: int = 60,
    horizon: int = 1,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    stride: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation DataLoaders from a DataFrame.

    Args:
        df: DataFrame with feature and target columns
        feature_cols: List of feature column names
        target_col: Target column name
        seq_len: Sequence length for windowing
        horizon: Forecast horizon
        train_ratio: Fraction of data for training
        batch_size: Batch size
        stride: Stride between windows
        num_workers: Number of DataLoader workers

    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    # Clean data
    cols = feature_cols + [target_col]
    df_clean = df.dropna(subset=cols).copy()

    features = df_clean[feature_cols].values.astype(np.float32)
    targets = df_clean[target_col].values.astype(np.float32)

    # Chronological split
    split_idx = int(len(features) * train_ratio)

    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]

    logger.info(
        f"Sequence dataset: train={len(train_features)}, val={len(val_features)}, "
        f"seq_len={seq_len}, horizon={horizon}"
    )

    train_dataset = TimeSeriesSequenceDataset(
        train_features, train_targets, seq_len, horizon, stride
    )
    val_dataset = TimeSeriesSequenceDataset(
        val_features, val_targets, seq_len, horizon, stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    info = {
        "n_features": len(feature_cols),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "seq_len": seq_len,
        "horizon": horizon,
        "feature_cols": feature_cols,
    }

    return train_loader, val_loader, info


def create_inference_sequence(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
) -> torch.Tensor:
    """
    Create a single input sequence from the latest data for inference.

    Args:
        df: DataFrame with feature columns
        feature_cols: Feature column names
        seq_len: Sequence length

    Returns:
        Tensor of shape (1, seq_len, n_features)
    """
    df_clean = df.dropna(subset=feature_cols)

    if len(df_clean) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rows for inference, got {len(df_clean)}"
        )

    features = df_clean[feature_cols].iloc[-seq_len:].values.astype(np.float32)
    return torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, n_features)
