"""Utilities for constructing and loading local Othello GPT checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import getpass
import os
import re

import torch

from othello_gpt import GPT, GPTConfig


DEFAULT_N_HEAD = 8


def get_local_dir(prefixes_to_resolve: list[str]) -> str:
    """Return a user-scoped cache directory from a list of candidate prefixes."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"

    prefix = prefixes_to_resolve[-1]
    os.makedirs(prefix, exist_ok=True)
    return f"{prefix}/{getpass.getuser()}"


def get_local_run_dir(exp_name: str, local_dirs: list[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def formatted_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in d.items()}


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a user-facing device string into a torch.device."""
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _strip_prefix_if_present(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> OrderedDict[str, torch.Tensor]:
    if not state_dict:
        return OrderedDict()
    if not all(key.startswith(prefix) for key in state_dict):
        return OrderedDict(state_dict.items())
    return OrderedDict((key[len(prefix) :], value) for key, value in state_dict.items())


def extract_state_dict(checkpoint: Any) -> OrderedDict[str, torch.Tensor]:
    """Extract a plain state dict from several common checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    candidate = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break

    state_dict = OrderedDict(candidate.items())
    for prefix in ("module.", "model."):
        state_dict = _strip_prefix_if_present(state_dict, prefix)
    return state_dict


def load_checkpoint_state_dict(
    checkpoint_path: str, map_location: str | torch.device = "cpu"
) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return extract_state_dict(checkpoint)


def infer_model_params_from_state_dict(
    state_dict: Mapping[str, torch.Tensor], n_head: int = DEFAULT_N_HEAD
) -> dict[str, int]:
    """Infer the GPT architecture directly from saved weights."""
    if "tok_emb.weight" not in state_dict or "pos_emb" not in state_dict:
        raise KeyError("Checkpoint must contain tok_emb.weight and pos_emb")

    vocab_size, n_embd = state_dict["tok_emb.weight"].shape
    _, block_size, pos_n_embd = state_dict["pos_emb"].shape
    if n_embd != pos_n_embd:
        raise ValueError(
            "Embedding size mismatch between tok_emb.weight and pos_emb: "
            f"{n_embd} vs {pos_n_embd}"
        )

    block_indices = [
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"blocks\.(\d+)\.", key))
    ]
    if not block_indices:
        raise ValueError("Could not infer n_layer from checkpoint keys")
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")

    return {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_layer": max(block_indices) + 1,
        "n_head": n_head,
    }


def load_model(config: Any) -> GPT:
    """
    Build a GPT model from explicit params or infer them from a checkpoint.

    Expected keys/attributes on ``config``:
    - ``device``: torch device, a device string, or ``"auto"``
    - ``params``: optional explicit GPT config params
    - ``model_path``: optional checkpoint path
    - ``n_head``: optional when inferring params from a checkpoint
    - ``load_weights``: whether to load checkpoint weights when ``model_path`` is set
    """

    device = resolve_device(_config_get(config, "device", "auto"))
    model_path = _config_get(config, "model_path")
    params = _config_get(config, "params")
    n_head = _config_get(config, "n_head", DEFAULT_N_HEAD)
    load_weights = bool(_config_get(config, "load_weights", True))

    state_dict = None
    if model_path:
        state_dict = load_checkpoint_state_dict(model_path, map_location="cpu")

    if params is None:
        if state_dict is None:
            raise ValueError("Provide either config.params or config.model_path")
        params = infer_model_params_from_state_dict(state_dict, n_head=n_head)

    model_config = GPTConfig(**params)
    model = GPT(model_config)
    model.config = model_config

    if state_dict is not None and load_weights:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def main(config: Any) -> None:
    """Keep the original training entrypoint importable without eager extra deps."""
    from omegaconf import OmegaConf
    import wandb

    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print(
            "Setting eval_every to",
            config.eval_every - config.eval_every % config.batch_size,
        )
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if config.sample_every % config.batch_size != 0:
        print("WARNING: sample_every must be divisible by batch_size")
        print(
            "Setting sample_every to",
            config.sample_every - config.sample_every % config.batch_size,
        )
        config.sample_every = (
            config.sample_every - config.sample_every % config.batch_size
        )

    config_path = os.path.join(config.local_run_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as file_p:
        OmegaConf.save(config, file_p)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    load_model(config.model)


if __name__ == "__main__":
    raise SystemExit(
        "This module exposes loading utilities. Use src/query_model.py for inference."
    )
