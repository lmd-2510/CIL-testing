from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MLPResNetForCIL(nn.Module):
    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        num_classes: int,
        device: torch.device,
        params: Dict[str, Any],
    ):
        super().__init__()
        self.categories = categories
        self.num_continuous = num_continuous
        self.num_classes = num_classes
        self.device = device
        self.params = params

        embed_dim = int(params.get("embed_dim", 16))
        hidden_dim = int(params.get("hidden_dim", 128))
        depth = int(params.get("depth", 3))
        dropout = float(params.get("dropout", 0.1))

        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(2, category_size), embed_dim) for category_size in categories]
        )

        input_dim = (len(categories) * embed_dim) + num_continuous
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(max(depth - 1, 0))]
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, cat_x: torch.Tensor, cont_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cat_x is None:
            batch_size = cont_x.shape[0] if cont_x is not None else 0
            cat_x = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        else:
            cat_x = cat_x.to(self.device, dtype=torch.long)

        if cont_x is None:
            cont_x = torch.empty((cat_x.shape[0], self.num_continuous), dtype=torch.float32, device=self.device)
        else:
            cont_x = cont_x.to(self.device, dtype=torch.float32)

        pieces = []
        if len(self.embeddings) > 0:
            embedded = [emb(cat_x[:, idx]) for idx, emb in enumerate(self.embeddings)]
            pieces.append(torch.cat(embedded, dim=1))
        if self.num_continuous > 0:
            pieces.append(cont_x)
        if not pieces:
            raise ValueError("Model received neither categorical nor continuous features.")

        x = torch.cat(pieces, dim=1)
        x = self.input_layer(x)
        x = self.residual_layers(x)
        return self.output_layer(x)


def _resolve_params(summary: Dict[str, Any]) -> Dict[str, Any]:
    config_defaults = summary.get("config_defaults", {})
    return {
        "embed_dim": config_defaults.get("mlp_embed_dim", 16),
        "hidden_dim": config_defaults.get("mlp_hidden_dim", 128),
        "depth": config_defaults.get("mlp_depth", 3),
        "dropout": config_defaults.get("mlp_dropout", 0.1),
    }


def build_model_for_cil(
    summary: Dict[str, Any],
    data_manager: Any,
    device: torch.device,
) -> MLPResNetForCIL:
    params = _resolve_params(summary)
    model = MLPResNetForCIL(
        categories=data_manager.categories,
        num_continuous=data_manager.num_continuous,
        num_classes=data_manager.num_classes,
        device=device,
        params=params,
    )
    return model.to(device)
