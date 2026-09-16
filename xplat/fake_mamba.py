"""CPU stand-in for ``mamba_ssm`` used only for smoke runs and tests.

``install()`` registers a module named ``mamba_ssm`` whose ``Mamba`` is a
causal depthwise convolution followed by a gated linear unit.  It has the
same constructor signature (``d_model``) and shape contract (batch, length,
d_model) as the real block, so the policy and point-process code paths run
end to end on any machine.  Numbers produced with it are meaningless.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn


class Mamba(nn.Module):
    def __init__(self, d_model, kernel=4, **_):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model)
        self.gate = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = nn.functional.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        h = self.conv(h).transpose(1, 2)
        a, b = self.gate(h).chunk(2, dim=-1)
        return self.out(a * torch.sigmoid(b))


def install(force=False):
    """Register the stand-in unless the real package imports (or ``force``)."""
    if not force:
        try:
            import mamba_ssm  # noqa: F401
            return False
        except Exception:
            pass
    mod = types.ModuleType("mamba_ssm")
    mod.Mamba = Mamba
    sys.modules["mamba_ssm"] = mod
    return True
