"""Residual-stream hooks — the one shared primitive `<h, v_hat>`.

Three modes over the same unit direction `v_hat` registered on a decoder layer:
  - read   (ProjectionReader): record <h, v_hat> per token; the monitor + audit path
  - add    (add_steering):     h += coef * v_hat; preventative / suppression mitigation
  - cap    (add_cap):          h -= v_hat * max(<h, v_hat> - tau, 0); bounded clamp

All hooks act on the layer's hidden-state output. `v_hat` must be unit-norm.
"""
from __future__ import annotations

import contextlib
import torch

# Global switch so the drift monitor can measure the *clean* weight-state of a steered
# run. add_steering / add_cap hooks no-op while paused. Without this, on_step measures
# activations through the active suppression hook → the steered arm's projection/probe
# trajectory is not comparable to the biased/neutral arms (it reads h-coef·v̂, not h).
_STEERING_ENABLED = True


@contextlib.contextmanager
def steering_paused():
    """Within this block, add_steering / add_cap hooks pass activations through unchanged."""
    global _STEERING_ENABLED
    prev = _STEERING_ENABLED
    _STEERING_ENABLED = False
    try:
        yield
    finally:
        _STEERING_ENABLED = prev


def _layer_module(model, layer: int):
    """Decoder block whose output is the residual stream at `layer`.

    Robust to both a plain HF CausalLM (`model.model.layers`) and a PEFT-wrapped model
    (an extra `.model` level): unwrap PEFT, then descend to whichever object holds
    `.layers` (Qwen/Llama layout). Override for other architectures.
    """
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    for holder in (getattr(m, "model", None),
                   getattr(getattr(m, "model", None), "model", None),
                   m):
        if holder is not None and hasattr(holder, "layers"):
            return holder.layers[layer]
    raise AttributeError(f"can't locate decoder layers on {type(model).__name__}")


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
        if not _STEERING_ENABLED:
            return out
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
        if not _STEERING_ENABLED:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = _as_tensor(v_hat, h)
        excess = torch.clamp((h @ v) - tau, min=0.0)    # (batch, seq)
        h = h - excess.unsqueeze(-1) * v
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return _layer_module(model, layer).register_forward_hook(_hook)
