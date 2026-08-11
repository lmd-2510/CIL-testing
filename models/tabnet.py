from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class GLUBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.norm = nn.BatchNorm1d(output_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.norm(self.linear(x)).chunk(2, dim=1)
        return self.dropout(value * torch.sigmoid(gate))


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers = [GLUBlock(input_dim, hidden_dim, dropout)]
        layers.extend(GLUBlock(hidden_dim, hidden_dim, dropout) for _ in range(max(depth - 1, 0)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, layer in enumerate(self.layers):
            transformed = layer(out)
            out = transformed if idx == 0 else 0.5 * (out + transformed)
        return out


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, feature_dim)
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, decision: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        scores = self.norm(self.proj(decision))
        return torch.softmax(scores * prior, dim=1)


class TabNetForCIL(nn.Module):
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

        embed_dim = int(params.get("embed_dim", 8))
        hidden_dim = int(params.get("hidden_dim", 64))
        n_steps = int(params.get("n_steps", 3))
        transformer_depth = int(params.get("transformer_depth", 2))
        dropout = float(params.get("dropout", 0.1))
        self.gamma = float(params.get("gamma", 1.5))

        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(2, category_size), embed_dim) for category_size in categories]
        )
        self.feature_dim = (len(categories) * embed_dim) + num_continuous
        self.n_steps = max(n_steps, 1)

        self.initial_transformer = FeatureTransformer(
            input_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            depth=transformer_depth,
            dropout=dropout,
        )
        self.step_transformers = nn.ModuleList(
            FeatureTransformer(self.feature_dim, hidden_dim, transformer_depth, dropout)
            for _ in range(self.n_steps)
        )
        self.attentive_transformers = nn.ModuleList(
            AttentiveTransformer(hidden_dim, self.feature_dim) for _ in range(self.n_steps)
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, cat_x: torch.Tensor, cont_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._merge_features(cat_x, cont_x)
        prior = torch.ones_like(x)
        decision = self.initial_transformer(x)
        aggregate = torch.zeros_like(decision)

        for step_idx in range(self.n_steps):
            mask = self.attentive_transformers[step_idx](decision, prior)
            masked_x = mask * x
            decision = self.step_transformers[step_idx](masked_x)
            aggregate = aggregate + torch.relu(decision)
            prior = prior * (self.gamma - mask)

        return self.output_layer(aggregate)

    def _merge_features(
        self,
        cat_x: Optional[torch.Tensor],
        cont_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
            pieces.append(torch.cat([emb(cat_x[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1))
        if self.num_continuous > 0:
            pieces.append(cont_x)
        if not pieces:
            raise ValueError("Model received neither categorical nor continuous features.")
        return torch.cat(pieces, dim=1)


def _resolve_params(summary: Dict[str, Any]) -> Dict[str, Any]:
    config_defaults = summary.get("config_defaults", {})
    return {
        "embed_dim": config_defaults.get("tabnet_embed_dim", 8),
        "hidden_dim": config_defaults.get("tabnet_hidden_dim", 64),
        "n_steps": config_defaults.get("tabnet_n_steps", 3),
        "transformer_depth": config_defaults.get("tabnet_transformer_depth", 2),
        "dropout": config_defaults.get("tabnet_dropout", 0.1),
        "gamma": config_defaults.get("tabnet_gamma", 1.5),
    }


def build_model_for_cil(
    summary: Dict[str, Any],
    data_manager: Any,
    device: torch.device,
) -> TabNetForCIL:
    params = _resolve_params(summary)
    model = TabNetForCIL(
        categories=data_manager.categories,
        num_continuous=data_manager.num_continuous,
        num_classes=data_manager.num_classes,
        device=device,
        params=params,
    )
    return model.to(device)
