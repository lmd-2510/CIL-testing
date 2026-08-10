from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWCMethod:
    """Elastic Weight Consolidation for class-incremental learning."""

    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        criterion: Optional[nn.Module] = None,
        fisher_n_batches: Optional[int] = None,
    ):
        self.ewc_lambda = ewc_lambda
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.fisher_n_batches = fisher_n_batches
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

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
        loss = loss + self._ewc_penalty(model)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def after_task(self, task_id: int, model: nn.Module, train_loader: DataLoader) -> None:
        fisher_estimate = self._estimate_fisher(model, train_loader)

        named_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        if not self.prev_params:
            self.prev_params = named_params
            self.fisher = fisher_estimate
            return None

        for name, fisher_value in fisher_estimate.items():
            if name in self.fisher:
                self.fisher[name] = self.fisher[name] + fisher_value
            else:
                self.fisher[name] = fisher_value
            self.prev_params[name] = named_params[name]

        return None

    def _ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.prev_params or not self.fisher:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.prev_params or name not in self.fisher:
                continue
            diff = param - self.prev_params[name]
            penalty = penalty + (self.fisher[name] * diff.pow(2)).sum()
        return 0.5 * self.ewc_lambda * penalty

    def _estimate_fisher(self, model: nn.Module, train_loader: DataLoader) -> Dict[str, torch.Tensor]:
        was_training = model.training
        model.eval()

        fisher = {
            name: torch.zeros_like(param, device=param.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if self.fisher_n_batches is not None and batch_idx >= self.fisher_n_batches:
                break

            cat_x, cont_x, targets = batch
            device = next(model.parameters()).device
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            targets = targets.to(device)

            model.zero_grad()
            logits = model(cat_x, cont_x)
            loss = self.criterion(logits, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher[name] += param.grad.detach().pow(2)

            num_batches += 1

        if num_batches > 0:
            for name in fisher:
                fisher[name] = fisher[name] / num_batches

        if was_training:
            model.train()

        return {
            name: value.detach().clone()
            for name, value in fisher.items()
        }


def build_method(
    ewc_lambda: float = 1000.0,
    criterion: Optional[nn.Module] = None,
    fisher_n_batches: Optional[int] = None,
) -> EWCMethod:
    return EWCMethod(
        ewc_lambda=ewc_lambda,
        criterion=criterion,
        fisher_n_batches=fisher_n_batches,
    )
