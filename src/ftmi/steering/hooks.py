"""Residual-stream hooks — the one shared primitive `<h, v_hat>`.

Three modes over the same unit direction `v_hat` registered on a decoder layer:
  - read   (ProjectionReader): record <h, v_hat> per token; the monitor + audit path
  - add    (add_steering):     h += coef * v_hat; preventative / suppression mitigation
  - cap    (add_cap):          h -= v_hat * max(<h, v_hat> - tau, 0); bounded clamp

All hooks act on the layer's hidden-state output. `v_hat` must be unit-norm.
"""
from __future__ import annotations

import torch


def _layer_module(model, layer: int):
    """Decoder block whose output is the residual stream at `layer`.

    Works for the HF Qwen/Llama layout (`model.model.layers[layer]`); override for
    other architectures.
    """
    return model.model.layers[layer]


def _as_tensor(v_hat, ref: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(v_hat, dtype=ref.dtype, device=ref.device)
    return v / v.norm()


class ProjectionReader:
    """Forward hook that accumulates per-token projections onto `v_hat`.

    Used for both training-time drift monitoring and inference-time detection
    (docs/vector-steering.md §3a). Read `.values` after a forward pass; call
    `.reset()` between units you want to score separately.
    """

    def __init__(self, model, layer: int, v_hat):
        self.values: list[torch.Tensor] = []
        self._v_hat = v_hat
        self._handle = _layer_module(model, layer).register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out  # (batch, seq, hidden)
        v = _as_tensor(self._v_hat, h) # unit-normalized concept direction
        self.values.append((h @ v).detach()) # ⟨h, v̂⟩ per token → (batch, seq)
        return out

    def reset(self) -> None:
        self.values.clear()

    def mean(self) -> float:
        if not self.values:
            return float("nan")
        return torch.cat([v.flatten() for v in self.values]).mean().item()

    def remove(self) -> None:
        self._handle.remove()


def add_steering(model, layer: int, v_hat, coef: float):
    """h <- h + coef * v_hat. Positive coef induces, negative suppresses.

    Registered during fine-tuning for preventative steering, or at decode for
    inference-time suppression. Returns the hook handle (call `.remove()`).
    """
    def _hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + coef * _as_tensor(v_hat, h)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return _layer_module(model, layer).register_forward_hook(_hook)


def add_cap(model, layer: int, v_hat, tau: float):
    """One-sided projection clamp: h <- h - v_hat * relu(<h, v_hat> - tau).

    No-ops when the projection is below tau. Bounds drift along the direction but
    does not reverse a weight-baked trait (controlled negative; Lu+ 2601.10387).
    Returns the hook handle.
    """
    def _hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = _as_tensor(v_hat, h)
        excess = torch.clamp((h @ v) - tau, min=0.0)    # (batch, seq)
        h = h - excess.unsqueeze(-1) * v
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return _layer_module(model, layer).register_forward_hook(_hook)
