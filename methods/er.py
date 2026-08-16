from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ERMethod:
    """Experience Replay baseline for class-incremental learning."""

    def __init__(
        self,
        criterion: Optional[nn.Module] = None,
        replay_weight: float = 1.0,
        replay_batch_size: Optional[int] = None,
    ):
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.replay_weight = float(replay_weight)
        self.replay_batch_size = replay_batch_size
        self.memory_batch: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        memory_batch: Tuple[torch.Tensor, ...],
    ) -> None:
        cat_x, cont_x, targets = memory_batch
        if len(targets) == 0:
            self.memory_batch = None
            return None
        self.memory_batch = (cat_x.clone(), cont_x.clone(), targets.clone())
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

        if self.memory_batch is not None and self.replay_weight > 0:
            mem_cat_x, mem_cont_x, mem_targets = self._sample_memory(
                fallback_batch_size=len(targets),
            )
            mem_cat_x = mem_cat_x.to(device)
            mem_cont_x = mem_cont_x.to(device)
            mem_targets = mem_targets.to(device)
            mem_logits = model(mem_cat_x, mem_cont_x)
            replay_loss = self.criterion(mem_logits, mem_targets)
            loss = loss + self.replay_weight * replay_loss

        loss.backward()
        optimizer.step()
        return float(loss.item())

    def after_task(self, task_id: int, model: nn.Module, train_loader: DataLoader) -> None:
        return None

    def _sample_memory(self, fallback_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.memory_batch is None:
            raise RuntimeError("Cannot sample replay memory before memory is initialized.")

        cat_x, cont_x, targets = self.memory_batch
        memory_size = len(targets)
        batch_size = self.replay_batch_size or fallback_batch_size
        batch_size = min(int(batch_size), memory_size)

        indices = torch.randint(0, memory_size, (batch_size,))
        return cat_x[indices], cont_x[indices], targets[indices]


def build_method(
    criterion: Optional[nn.Module] = None,
    replay_weight: float = 1.0,
    replay_batch_size: Optional[int] = None,
) -> ERMethod:
    return ERMethod(
        criterion=criterion,
        replay_weight=replay_weight,
        replay_batch_size=replay_batch_size,
    )
