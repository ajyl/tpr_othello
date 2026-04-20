from typing import Callable, Dict, Union, Any
from contextlib import contextmanager
import torch
from torch import nn
from hook_utils.record_utils import get_module, untuple_tensor

OverrideSpec = Union[
    torch.Tensor,
    Callable[[Any, tuple, nn.Module, str], Any],
]


def _materialize_tensor_override(spec: torch.Tensor, output: Any, name: str) -> torch.Tensor:
    reference = untuple_tensor(output) if isinstance(output, (tuple, list)) else output
    if not isinstance(reference, torch.Tensor):
        raise TypeError(
            f"Expected tensor-like output at {name}, got {type(reference)} instead."
        )

    override = spec.to(device=reference.device, dtype=reference.dtype)
    if override.shape != reference.shape:
        try:
            override = torch.broadcast_to(override, reference.shape)
        except RuntimeError as exc:
            raise AssertionError(
                f"Shape mismatch in intervention at {name}: "
                f"expected broadcastable to {reference.shape}, got {override.shape}"
            ) from exc
    return override


@contextmanager
def intervene(
    model: nn.Module,
    overrides: Dict[str, OverrideSpec],
):
    """
    Context manager for interventions.

    Args:
        model: Hooked model.
        overrides: dict mapping module_name -> spec, where spec is either:
            - a tensor: this tensor will be returned as the output of that module
              (must be broadcastable / same shape as original output), or
            - a function(orig_output, inputs, module, name) -> new_output
              giving full control over how to transform the output.
    """
    handles = []

    for name, spec in overrides.items():
        module = get_module(model, name)

        def make_hook(name: str, spec: OverrideSpec):
            def hook(_module: nn.Module, _input: tuple, _output: Any):
                if callable(spec):
                    return spec(_output, _input, _module, name)

                if isinstance(_output, torch.Tensor):
                    return _materialize_tensor_override(spec, _output, name)

                if isinstance(_output, (tuple, list)):
                    override = _materialize_tensor_override(spec, _output, name)
                    updated = [override, *_output[1:]]
                    return tuple(updated) if isinstance(_output, tuple) else updated

                raise TypeError(
                    f"Unsupported output type at {name}: {type(_output)}"
                )

            return hook

        hook_fn = make_hook(name, spec)
        h = module.register_forward_hook(hook_fn)
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()
