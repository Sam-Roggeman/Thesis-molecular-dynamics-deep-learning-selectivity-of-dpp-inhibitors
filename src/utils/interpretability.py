from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from captum.attr import GuidedBackprop, Occlusion
from captum.attr import IntegratedGradients
from captum.attr import LayerIntegratedGradients
from captum.attr import Saliency
from captum.attr import GradientShap
from torch import nn


@dataclass
class AttributionResult:
    """Container for attribution outputs returned by Captum methods."""

    attributions: torch.Tensor
    method: str
    target: int | torch.Tensor | None = None
    convergence_delta: torch.Tensor | None = None


class CaptumInterpreter:
    """Utility wrapper that adds Captum interpretability methods to any PyTorch model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @staticmethod
    def summarize_attributions(
        attributions: torch.Tensor,
        reduce_dim: int | tuple[int, ...] | None = 1,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Reduce and optionally normalize attribution magnitudes for plotting or inspection."""
        summarized = attributions.abs()
        if reduce_dim is not None:
            summarized = summarized.sum(dim=reduce_dim)

        if normalize:
            denom = summarized.amax(dim=tuple(range(1, summarized.ndim)), keepdim=True)
            summarized = summarized / (denom + 1e-12)

        return summarized

    def _resolve_target(self, outputs: torch.Tensor, target: int | torch.Tensor | None) -> int | torch.Tensor:
        if target is not None:
            return target
        if outputs.ndim == 1:
            return 0
        return outputs.argmax(dim=1)

    def _prepare_model(self) -> None:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor | None = None,
        baselines: torch.Tensor | None = None,
        n_steps: int = 50,
        additional_forward_args: Any | None = None,
        internal_batch_size: int | None = None,
        return_convergence_delta: bool = True,
    ) -> AttributionResult:
        self._prepare_model()
        ig = IntegratedGradients(self.model)

        with torch.enable_grad():
            outputs = self.model(inputs)
            resolved_target = self._resolve_target(outputs, target)

            if return_convergence_delta:
                attributions, delta = ig.attribute(
                    inputs,
                    baselines=baselines,
                    target=resolved_target,
                    additional_forward_args=additional_forward_args,
                    n_steps=n_steps,
                    internal_batch_size=internal_batch_size,
                    return_convergence_delta=True,
                )
                return AttributionResult(
                    attributions=attributions,
                    method="integrated_gradients",
                    target=resolved_target,
                    convergence_delta=delta,
                )

            attributions = ig.attribute(
                inputs,
                baselines=baselines,
                target=resolved_target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
            )
            return AttributionResult(
                attributions=attributions,
                method="integrated_gradients",
                target=resolved_target,
            )

    def saliency(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor | None = None,
        additional_forward_args: Any | None = None,
        abs_values: bool = True,
    ) -> AttributionResult:
        self._prepare_model()
        saliency = Saliency(self.model)

        with torch.enable_grad():
            outputs = self.model(inputs)
            resolved_target = self._resolve_target(outputs, target)
            attributions = saliency.attribute(
                inputs,
                target=resolved_target,
                additional_forward_args=additional_forward_args,
                abs=abs_values,
            )

        return AttributionResult(
            attributions=attributions,
            method="saliency",
            target=resolved_target,
        )

    def occlusion(self, inputs: torch.Tensor, target: int | torch.Tensor | None = None, patch_size: int = 1, shift_size: int = 1, perturbations_per_eval: int = 10) -> AttributionResult:
        self._prepare_model()
        ablator = Occlusion(self.model)
        with torch.enable_grad():
            outputs = self.model(inputs)
        resolved_target = self._resolve_target(outputs, target)
        # Computes occlusion attribution, ablating each patch_size x patch_size patch
        # shifting in each direction by the default of 1.
        attributions = ablator.attribute(inputs, target=resolved_target, sliding_window_shapes=(1, patch_size, patch_size), strides=(1, shift_size, shift_size), perturbations_per_eval=perturbations_per_eval)
        return AttributionResult(
            attributions=attributions,
            method="occlusion",
            target=resolved_target,
        )