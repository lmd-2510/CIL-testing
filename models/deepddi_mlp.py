from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class DeepDDIMLPForCIL(nn.Module):
    """DeepDDI-style descriptor MLP adapted for CIL.

    The original DeepDDI family predicts DDI events from drug-pair vector
    representations. In this repo, the drug-pair representation is the
    precomputed tabular feature vector: encoded drug metadata plus QSAR
    descriptors.
    """

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

        embed_dim = int(params.get("embed_dim", 16))
        descriptor_dim = int(params.get("descriptor_dim", 512))
        hidden_dim = int(params.get("hidden_dim", 512))
        depth = int(params.get("depth", 3))
        dropout = float(params.get("dropout", 0.3))
        use_batch_norm = bool(params.get("batch_norm", True))

        self.embeddings = nn.ModuleList(
            [nn.Embedding(max(2, category_size), embed_dim) for category_size in categories]
        )

        cat_dim = len(categories) * embed_dim
        self.continuous_encoder = nn.Sequential(
            nn.Linear(num_continuous, descriptor_dim),
            nn.BatchNorm1d(descriptor_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        ) if num_continuous > 0 else nn.Identity()

        input_dim = cat_dim + (descriptor_dim if num_continuous > 0 else 0)
        layers = []
        for layer_idx in range(max(depth, 1)):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.classifier = nn.Sequential(*layers, nn.Linear(hidden_dim, num_classes))

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
            pieces.append(torch.cat([emb(cat_x[:, idx]) for idx, emb in enumerate(self.embeddings)], dim=1))
        if self.num_continuous > 0:
            pieces.append(self.continuous_encoder(cont_x))
        if not pieces:
            raise ValueError("Model received neither categorical nor continuous features.")

        return self.classifier(torch.cat(pieces, dim=1))


def _resolve_params(summary: Dict[str, Any]) -> Dict[str, Any]:
    config_defaults = summary.get("config_defaults", {})
    return {
        "embed_dim": config_defaults.get("deepddi_embed_dim", 16),
        "descriptor_dim": config_defaults.get("deepddi_descriptor_dim", 512),
        "hidden_dim": config_defaults.get("deepddi_hidden_dim", 512),
        "depth": config_defaults.get("deepddi_depth", 3),
        "dropout": config_defaults.get("deepddi_dropout", 0.3),
        "batch_norm": config_defaults.get("deepddi_batch_norm", True),
    }


def build_model_for_cil(
    summary: Dict[str, Any],
    data_manager: Any,
    device: torch.device,
) -> DeepDDIMLPForCIL:
    params = _resolve_params(summary)
    model = DeepDDIMLPForCIL(
        categories=data_manager.categories,
        num_continuous=data_manager.num_continuous,
        num_classes=data_manager.num_classes,
        device=device,
        params=params,
    )
    return model.to(device)
