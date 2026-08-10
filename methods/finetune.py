from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FinetuneMethod:
    """Naive sequential fine-tuning baseline for class-incremental learning."""

    def __init__(self, criterion: Optional[nn.Module] = None):
        self.criterion = criterion or nn.CrossEntropyLoss()

    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        memory_batch: Tuple[torch.Tensor, ...],
    ) -> None:
        return None

    def training_step(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        cat_x, cont_x, targets = batch
        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(cat_x, cont_x)
        loss = self.criterion(logits, targets)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def after_task(self, task_id: int, model: nn.Module, train_loader: DataLoader) -> None:
        return None


def build_method(criterion: Optional[nn.Module] = None) -> FinetuneMethod:
    return FinetuneMethod(criterion=criterion)
