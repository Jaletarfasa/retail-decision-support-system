from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


def _set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this to be configured only before interop work starts.
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class _TrainingConfig:
    hidden_dims: tuple[int, ...]
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int


class _MLPRegressorModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class _EntityEmbeddingRegressorModule(nn.Module):
    def __init__(
        self,
        embedding_sizes: list[tuple[int, int]],
        n_continuous: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality, embedding_dim in embedding_sizes]
        )
        input_dim = sum(embedding_dim for _, embedding_dim in embedding_sizes) + n_continuous
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.embeddings:
            parts.extend(embedding(categorical[:, idx]) for idx, embedding in enumerate(self.embeddings))
        if continuous.shape[1] > 0:
            parts.append(continuous)
        features = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.network(features).squeeze(-1)


class TabularMLPRegressor:
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
        learning_rate: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.config = _TrainingConfig(hidden_dims, learning_rate, epochs, batch_size, seed)
        self.scaler = StandardScaler()
        self.model: _MLPRegressorModule | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularMLPRegressor":
        _set_reproducible_seed(self.config.seed)
        self.feature_names_ = list(X.columns)
        X_train = self.scaler.fit_transform(X[self.feature_names_].astype(float))
        y_train = y.astype(float).to_numpy(dtype=np.float32)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.model = _MLPRegressorModule(X_tensor.shape[1], self.config.hidden_dims)
        self._train_loop(X_tensor, y_tensor)
        return self

    def _train_loop(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        assert self.model is not None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        n_rows = features.shape[0]
        batch_size = max(1, min(self.config.batch_size, n_rows))
        rng = np.random.default_rng(self.config.seed)
        self.model.train()
        for _ in range(self.config.epochs):
            indices = rng.permutation(n_rows)
            for start in range(0, n_rows, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_features = features[batch_idx]
                batch_targets = targets[batch_idx]
                optimizer.zero_grad()
                preds = self.model(batch_features)
                loss = loss_fn(preds, batch_targets)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fit before predict.")
        X_eval = self.scaler.transform(X[self.feature_names_].astype(float))
        X_tensor = torch.tensor(X_eval, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()


class EntityEmbeddingRegressor:
    def __init__(
        self,
        categorical_cols: tuple[str, ...] = ("store_id", "sku_id"),
        hidden_dims: tuple[int, ...] = (64, 32),
        learning_rate: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.categorical_cols = categorical_cols
        self.config = _TrainingConfig(hidden_dims, learning_rate, epochs, batch_size, seed)
        self.continuous_scaler = StandardScaler()
        self.category_maps_: dict[str, dict[object, int]] = {}
        self.feature_names_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.continuous_cols_: list[str] = []
        self.model: _EntityEmbeddingRegressorModule | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EntityEmbeddingRegressor":
        _set_reproducible_seed(self.config.seed)
        self.feature_names_ = list(X.columns)
        self.categorical_cols_ = [col for col in self.categorical_cols if col in self.feature_names_]
        self.continuous_cols_ = [col for col in self.feature_names_ if col not in self.categorical_cols_]

        categorical_tensor, embedding_sizes = self._encode_categoricals(X, fit=True)
        continuous_tensor = self._encode_continuous(X, fit=True)
        targets = torch.tensor(y.astype(float).to_numpy(dtype=np.float32), dtype=torch.float32)
        self.model = _EntityEmbeddingRegressorModule(
            embedding_sizes=embedding_sizes,
            n_continuous=continuous_tensor.shape[1],
            hidden_dims=self.config.hidden_dims,
        )
        self._train_loop(categorical_tensor, continuous_tensor, targets)
        return self

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if not self.categorical_cols_:
            empty = torch.empty((len(X), 0), dtype=torch.long)
            return empty, []

        encoded_cols: list[np.ndarray] = []
        embedding_sizes: list[tuple[int, int]] = []
        for col in self.categorical_cols_:
            series = X[col]
            if fit:
                unique_values = sorted(series.dropna().unique().tolist())
                self.category_maps_[col] = {value: idx + 1 for idx, value in enumerate(unique_values)}
            mapping = self.category_maps_[col]
            encoded = series.map(mapping).fillna(0).astype(int).to_numpy()
            encoded_cols.append(encoded)
            cardinality = len(mapping) + 1
            embedding_dim = min(32, max(4, (cardinality + 1) // 2))
            embedding_sizes.append((cardinality, embedding_dim))

        stacked = np.column_stack(encoded_cols)
        return torch.tensor(stacked, dtype=torch.long), embedding_sizes

    def _encode_continuous(self, X: pd.DataFrame, fit: bool) -> torch.Tensor:
        if not self.continuous_cols_:
            return torch.empty((len(X), 0), dtype=torch.float32)
        continuous = X[self.continuous_cols_].astype(float)
        values = self.continuous_scaler.fit_transform(continuous) if fit else self.continuous_scaler.transform(continuous)
        return torch.tensor(values, dtype=torch.float32)

    def _train_loop(
        self,
        categorical_tensor: torch.Tensor,
        continuous_tensor: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        assert self.model is not None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        n_rows = targets.shape[0]
        batch_size = max(1, min(self.config.batch_size, n_rows))
        rng = np.random.default_rng(self.config.seed)
        self.model.train()
        for _ in range(self.config.epochs):
            indices = rng.permutation(n_rows)
            for start in range(0, n_rows, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_cat = categorical_tensor[batch_idx]
                batch_cont = continuous_tensor[batch_idx]
                batch_targets = targets[batch_idx]
                optimizer.zero_grad()
                preds = self.model(batch_cat, batch_cont)
                loss = loss_fn(preds, batch_targets)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fit before predict.")
        categorical_tensor, _ = self._encode_categoricals(X, fit=False)
        continuous_tensor = self._encode_continuous(X, fit=False)
        self.model.eval()
        with torch.no_grad():
            return self.model(categorical_tensor, continuous_tensor).cpu().numpy()
