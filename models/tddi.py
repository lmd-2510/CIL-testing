from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tab_transformer_pytorch import TabTransformer

    TAB_TRANSFORMER_AVAILABLE = True
except ImportError:
    TabTransformer = None
    TAB_TRANSFORMER_AVAILABLE = False

try:
    from utils import UncertaintyEstimator
except ImportError:
    from .utils import UncertaintyEstimator


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


class SimpleTabularBackbone(nn.Module):
    """Fallback backbone when tab_transformer_pytorch is unavailable."""

    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int,
        dim_out: int,
        depth: int,
        ff_dropout: float,
    ):
        super().__init__()
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dim = dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(2, category_size), dim) for category_size in categories]
        )
        input_dim = (self.num_categories * dim) + num_continuous

        layers: List[nn.Module] = []
        hidden_dim = max(dim, 32)
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if ff_dropout > 0:
                layers.append(nn.Dropout(ff_dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_x: torch.Tensor, cont_x: torch.Tensor) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        if self.num_categories > 0:
            embedded = [emb(cat_x[:, idx]) for idx, emb in enumerate(self.embeddings)]
            pieces.append(torch.cat(embedded, dim=1))
        if self.num_continuous > 0:
            pieces.append(cont_x)
        if not pieces:
            raise ValueError("Model received neither categorical nor continuous features.")
        return self.mlp(torch.cat(pieces, dim=1))


class TDDIForCIL(nn.Module):
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        num_classes: int,
        device: torch.device,
        best_params: Dict[str, Any],
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.num_classes = num_classes
        self.device = device
        self.best_params = best_params
        self.uncertainty_estimator = UncertaintyEstimator()

        self.backbone = self._build_backbone().to(self.device)

    def _build_backbone(self) -> nn.Module:
        dim = int(self.best_params.get("dim", 64))
        depth = int(self.best_params.get("depth", 3))
        ff_dropout = float(self.best_params.get("ff_dropout", 0.1))

        if TAB_TRANSFORMER_AVAILABLE:
            return TabTransformer(
                categories=tuple(self.categories),
                num_continuous=self.num_continuous,
                dim=dim,
                dim_out=self.num_classes,
                depth=depth,
                heads=int(self.best_params.get("heads", 8)),
                attn_dropout=float(self.best_params.get("attn_dropout", ff_dropout)),
                ff_dropout=ff_dropout,
                mlp_hidden_mults=(
                    int(self.best_params.get("mlp_hidden_mult_1", 4)),
                    int(self.best_params.get("mlp_hidden_mult_2", 2)),
                ),
                mlp_act=nn.ReLU(),
            )

        return SimpleTabularBackbone(
            categories=self.categories,
            num_continuous=self.num_continuous,
            dim=dim,
            dim_out=self.num_classes,
            depth=depth,
            ff_dropout=ff_dropout,
        )

    def forward(self, cat_x: torch.Tensor, cont_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cat_x is None:
            batch_size = cont_x.shape[0] if cont_x is not None else 0
            cat_x = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        else:
            cat_x = cat_x.to(self.device, dtype=torch.long)

        if cont_x is None:
            batch_size = cat_x.shape[0]
            cont_x = torch.empty(
                (batch_size, self.num_continuous), dtype=torch.float32, device=self.device
            )
        else:
            cont_x = cont_x.to(self.device, dtype=torch.float32)

        return self.backbone(cat_x, cont_x)

    def predict_batch(
        self, cat_x: torch.Tensor, cont_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(cat_x, cont_x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    def predict_with_uncertainty(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        features = test_df.drop(columns=["class"], errors="ignore")
        feature_tensor = torch.tensor(features.values, dtype=torch.float32)
        dataset = TensorDataset(feature_tensor, torch.zeros(len(feature_tensor)))
        test_loader = DataLoader(
            dataset,
            batch_size=int(self.best_params.get("batch_size", 256)),
            shuffle=False,
        )

        predictions: List[np.ndarray] = []
        probabilities: List[np.ndarray] = []
        self.eval()
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                cat_x = (
                    batch_x[:, : len(self.categories)].long()
                    if len(self.categories) > 0
                    else torch.empty((batch_x.shape[0], 0), dtype=torch.long, device=self.device)
                )
                cont_x = (
                    batch_x[:, len(self.categories) :]
                    if self.num_continuous > 0
                    else torch.empty((batch_x.shape[0], 0), dtype=torch.float32, device=self.device)
                )
                logits = self.forward(cat_x, cont_x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                predictions.append(preds.cpu().numpy())
                probabilities.append(probs.cpu().numpy())

        merged_predictions = np.concatenate(predictions, axis=0) if predictions else np.array([])
        merged_probabilities = (
            np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, self.num_classes))
        )
        uncertainties = self.uncertainty_estimator.entropy_uncertainty(merged_probabilities)
        return {
            "predictions": merged_predictions,
            "probabilities": merged_probabilities,
            "uncertainties": uncertainties,
        }


class TDDI_Model:
    """Compatibility wrapper around the trainable CIL model."""

    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        num_classes: int,
        device: torch.device,
        best_params: Dict[str, Any],
    ):
        self.categories = categories
        self.num_continuous = num_continuous
        self.num_classes = num_classes
        self.device = device
        self.best_params = best_params
        self.model = TDDIForCIL(
            categories=categories,
            num_continuous=num_continuous,
            num_classes=num_classes,
            device=device,
            best_params=best_params,
        )

    def create_model(self) -> TDDIForCIL:
        return self.model


def _resolve_params(summary: Dict[str, Any]) -> Dict[str, Any]:
    config_defaults = summary.get("config_defaults", {})
    dim = config_defaults.get("dim", config_defaults.get("hidden_dim", 64))
    return {
        "dim": dim,
        "depth": config_defaults.get("depth", 3),
        "heads": config_defaults.get("heads", 8),
        "attn_dropout": config_defaults.get(
            "attn_dropout", config_defaults.get("ff_dropout", 0.1)
        ),
        "ff_dropout": config_defaults.get("ff_dropout", 0.1),
        "mlp_hidden_mult_1": config_defaults.get("mlp_hidden_mult_1", 4),
        "mlp_hidden_mult_2": config_defaults.get("mlp_hidden_mult_2", 2),
        "batch_size": config_defaults.get("batch_size", 256),
        "focal_gamma": config_defaults.get("focal_gamma", 1.0),
    }


def build_model_for_cil(summary: Dict[str, Any], data_manager: Any, device: torch.device) -> TDDIForCIL:
    params = _resolve_params(summary)
    model = TDDIForCIL(
        categories=data_manager.categories,
        num_continuous=data_manager.num_continuous,
        num_classes=data_manager.num_classes,
        device=device,
        best_params=params,
    )
    return model.to(device)
