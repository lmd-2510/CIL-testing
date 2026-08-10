from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AGEMMethod:
    """Averaged Gradient Episodic Memory for class-incremental learning."""

    def __init__(
        self,
        criterion: Optional[nn.Module] = None,
    ):
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.reference_memory: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def before_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        memory_batch: Tuple[torch.Tensor, ...],
    ) -> None:
        cat_x, cont_x, targets = memory_batch
        if len(targets) == 0:
            self.reference_memory = None
            return None
        self.reference_memory = (cat_x.clone(), cont_x.clone(), targets.clone())
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

        current_loss = self._compute_loss(model, cat_x, cont_x, targets)
        current_grad = self._compute_gradient_from_loss(model, current_loss)
        final_grad = current_grad

        if self.reference_memory is not None:
            ref_cat_x, ref_cont_x, ref_targets = self.reference_memory
            ref_cat_x = ref_cat_x.to(device)
            ref_cont_x = ref_cont_x.to(device)
            ref_targets = ref_targets.to(device)

            reference_loss = self._compute_loss(model, ref_cat_x, ref_cont_x, ref_targets)
            reference_grad = self._compute_gradient_from_loss(model, reference_loss)

            if self._has_conflict(current_grad, reference_grad):
                final_grad = self._project_gradient(current_grad, reference_grad)

        optimizer.zero_grad()
        self._set_gradient(model, final_grad)
        optimizer.step()
        return float(current_loss.item())

    def after_task(self, task_id: int, model: nn.Module, train_loader: DataLoader) -> None:
        return None

    def _compute_loss(
        self,
        model: nn.Module,
        cat_x: torch.Tensor,
        cont_x: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = model(cat_x, cont_x)
        return self.criterion(logits, targets)

    def _compute_gradient_from_loss(
        self,
        model: nn.Module,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        model.zero_grad()
        loss.backward()
        return self._flatten_gradients(model)

    def _flatten_gradients(self, model: nn.Module) -> torch.Tensor:
        grads: List[torch.Tensor] = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                grads.append(torch.zeros(param.numel(), device=param.device))
            else:
                grads.append(param.grad.detach().flatten().clone())
        if not grads:
            return torch.empty(0)
        return torch.cat(grads)

    def _set_gradient(self, model: nn.Module, flat_grad: torch.Tensor) -> None:
        pointer = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            grad_slice = flat_grad[pointer : pointer + numel].view_as(param)
            param.grad = grad_slice.clone()
            pointer += numel

    def _has_conflict(self, current_grad: torch.Tensor, reference_grad: torch.Tensor) -> bool:
        if current_grad.numel() == 0 or reference_grad.numel() == 0:
            return False
        return torch.dot(current_grad, reference_grad).item() < 0

    def _project_gradient(
        self,
        current_grad: torch.Tensor,
        reference_grad: torch.Tensor,
    ) -> torch.Tensor:
        denominator = torch.dot(reference_grad, reference_grad)
        if denominator.item() <= 0:
            return current_grad

        dot_product = torch.dot(current_grad, reference_grad)
        if dot_product.item() >= 0:
            return current_grad

        correction = (dot_product / denominator) * reference_grad
        return current_grad - correction


def build_method(
    criterion: Optional[nn.Module] = None,
) -> AGEMMethod:
    return AGEMMethod(criterion=criterion)
