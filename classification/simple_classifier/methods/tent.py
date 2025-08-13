from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tent(nn.Module):
    """
    Test-Time Entropy Minimization (TENT) — standalone wrapper.

    - No dependency on TTAMethod or registries.
    - Configures the wrapped model like the original TENT paper/code.
    - Updates only norm-layer affine params by default (weight/bias).

    Parameters
    ----------
    model : nn.Module
        Pretrained classifier that outputs logits (B, C).
    optimizer : torch.optim.Optimizer, optional
        If None, Adam on norm-layer affine params is created.
    device : str, optional
        "cuda" or "cpu". Defaults to available device.
    mixed_precision : bool
        Enable AMP (CUDA only).
    lr : float
        Learning rate if creating a default optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | None = None,
        mixed_precision: bool = False,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.original_model = deepcopy(model)
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = bool(mixed_precision) and self.device.startswith("cuda")

        # Configure wrapped model as in Tent
        self.configure_model()

        # Default optimizer over norm affine params
        if optimizer is None:
            params = self.collect_norm_params(self.model)
            if not params:  # fallback if no norms are present
                params = list(self.model.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)
        self.optimizer = optimizer

        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Move to device
        self.to(self.device)

    # ----------------- Core API -----------------
    def loss_calculation(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device, non_blocking=True)
        logits = self.model(x)
        loss = self.softmax_entropy(logits).mean()
        return logits, loss

    @torch.enable_grad()
    def forward_and_adapt(self, batch):
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            outputs, loss = self.loss_calculation(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return outputs

    # Convenience alias
    step = forward_and_adapt

    # ----------------- Helpers -----------------
    @staticmethod
    def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Per-sample softmax entropy: H(p) = -∑ p log p."""
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        return -(p * logp).sum(dim=1)

    @staticmethod
    def collect_norm_params(model: nn.Module):
        params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for name, p in m.named_parameters(recurse=False):
                    if name in ("weight", "bias"):
                        params.append(p)
        return params

    def configure_model(self) -> None:
        """Mirror Tent config: eval mode; enable grads on norm affine params.
        BN2d uses batch stats (track_running_stats=False) even in eval.
        BN1d kept in train() to handle single-sample batches.
        """
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)