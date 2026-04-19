"""Train Smolensky-style tensor-product board probes on Othello GPT activations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "hook_utils"))

from load_model import load_model  # noqa: E402
from hook_utils import convert_to_hooked_model, record_activations, seed_all  # noqa: E402
from train_probe import (  # noqa: E402
    STARTING_SQUARES,
    BOARD_COLS,
    BOARD_LABEL_OPTIONS,
    BOARD_ROWS,
    build_state_stack,
    format_square_accuracy_board,
    load_probe_dataset,
    macro_f1_score,
    resolve_position_slice,
    select_layers,
    state_stack_to_one_hot_threeway,
)


class TensorProductBoardProbe(torch.nn.Module):
    """
    Probe that decodes board state through a role x filler binding tensor.

    For each residual vector h_t, the probe learns a linear map into a binding
    tensor T_t in role/filler space. Square/class logits are then computed by
    unbinding T_t with learned square-role and filler vectors:

        T_t = sum_d h_t[d] * U[d]
        logit[row, col, class] = role[row, col]^T T_t filler[class]
    """

    def __init__(
        self,
        d_model: int,
        role_dim: int,
        filler_dim: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.role_dim = int(role_dim)
        self.filler_dim = int(filler_dim)
        self.use_bias = bool(use_bias)

        self.binding_map = torch.nn.Parameter(
            torch.randn(d_model, role_dim, filler_dim) / math.sqrt(d_model)
        )
        self.role_embeddings = torch.nn.Parameter(
            torch.randn(BOARD_ROWS, BOARD_COLS, role_dim) / math.sqrt(role_dim)
        )
        self.filler_embeddings = torch.nn.Parameter(
            torch.randn(BOARD_LABEL_OPTIONS, filler_dim) / math.sqrt(filler_dim)
        )
        if self.use_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(BOARD_ROWS, BOARD_COLS, BOARD_LABEL_OPTIONS)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, residual: Tensor) -> Tensor:
        binding_tensor = torch.einsum("bpd,drf->bprf", residual, self.binding_map)
        logits = torch.einsum(
            "bprf,xyr,cf->bpxyc",
            binding_tensor,
            self.role_embeddings,
            self.filler_embeddings,
        )
        if self.bias is not None:
            logits = logits + self.bias
        return logits


DEFAULT_TPR_ACTIVATION_NAME = "hook_resid_post"
TPR_ACTIVATION_NAME_ALIASES = {
    "resid_post": "hook_resid_post",
    "hook_resid_post": "hook_resid_post",
    "resid_mid": "hook_resid_mid",
    "hook_resid_mid": "hook_resid_mid",
}
DEFAULT_BINDING_TO_RESIDUAL_RIDGE = 1e-4


def normalize_activation_name(name: str) -> str:
    try:
        return TPR_ACTIVATION_NAME_ALIASES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(TPR_ACTIVATION_NAME_ALIASES))
        raise ValueError(f"Unsupported activation name {name!r}. Expected one of: {valid}") from exc


def activation_name_to_tag(name: str) -> str:
    normalized_name = normalize_activation_name(name)
    if normalized_name == "hook_resid_post":
        return "resid"
    if normalized_name == "hook_resid_mid":
        return "resid_mid"
    raise ValueError(f"Unsupported normalized activation name: {normalized_name}")


def make_module_name(layer: int, activation_name: str) -> str:
    return f"blocks.{layer}.{normalize_activation_name(activation_name)}"


def infer_activation_name_from_artifact(artifact: dict) -> str:
    if "activation_name" in artifact:
        return normalize_activation_name(str(artifact["activation_name"]))

    module_name = artifact.get("module_name")
    if isinstance(module_name, str):
        activation_name = module_name.rsplit(".", 1)[-1]
        return normalize_activation_name(activation_name)

    config = artifact.get("config")
    if isinstance(config, dict) and "activation_name" in config:
        return normalize_activation_name(str(config["activation_name"]))

    return DEFAULT_TPR_ACTIVATION_NAME


@dataclass
class TPRProbeConfig:
    checkpoint: str = "ckpts/synthetic_model.pth"
    data_path: str = "train_data"
    output_dir: str = "probes/tpr"
    device: str = "auto"
    random_init: bool = False
    n_head: int = 8
    lr: float = 1e-2
    wd: float = 1e-2
    batch_size: int = 128
    valid_every: int = 100
    pos_start: int = 0
    pos_end: int | None = None
    num_epochs: int = 1
    valid_size: int = 512
    test_size: int = 1000
    valid_patience: int = 10
    seed: int = 1111
    max_games: int | None = None
    max_layers: int | None = None
    train_layers: str | None = "6"
    role_dim: int = 16
    filler_dim: int = 8
    use_bias: bool = False
    exclude_center_squares: bool = False
    activation_name: str = DEFAULT_TPR_ACTIVATION_NAME
    binding_to_residual_ridge: float = DEFAULT_BINDING_TO_RESIDUAL_RIDGE


def compute_tpr_probe_logits(residual: Tensor, probe: TensorProductBoardProbe) -> Tensor:
    return probe(residual)


def compute_tpr_binding_tensor(
    residual: Tensor,
    binding_map: Tensor,
) -> Tensor:
    return torch.einsum("bpd,drf->bprf", residual, binding_map)


def load_binding_to_residual_linear_map_from_artifact(
    artifact: dict,
    device: torch.device | None = None,
) -> dict | None:
    saved_map = artifact.get("binding_to_residual")
    if not isinstance(saved_map, dict):
        return None

    weight = saved_map.get("weight")
    if not isinstance(weight, torch.Tensor):
        return None

    bias = saved_map.get("bias")
    loaded_map = dict(saved_map)
    loaded_map["weight"] = weight.to(device) if device is not None else weight
    loaded_map["bias"] = (
        bias.to(device)
        if isinstance(bias, torch.Tensor) and device is not None
        else bias
        if isinstance(bias, torch.Tensor)
        else None
    )
    return loaded_map


def build_square_selection_mask(
    *,
    exclude_center_squares: bool,
    device: torch.device | str,
) -> Tensor:
    mask = torch.ones(BOARD_ROWS, BOARD_COLS, device=device, dtype=torch.bool)
    if exclude_center_squares:
        for board_pos in STARTING_SQUARES:
            row_idx, col_idx = divmod(board_pos, BOARD_COLS)
            mask[row_idx, col_idx] = False
    return mask


def compute_tpr_probe_loss(
    probe_logits: Tensor,
    state_stack_one_hot: Tensor,
    square_mask: Tensor,
) -> Tensor:
    log_probs = probe_logits.log_softmax(dim=-1)
    correct_log_probs = (log_probs * state_stack_one_hot).sum(dim=-1).mean(
        dim=0
    ) * BOARD_LABEL_OPTIONS
    selected_correct_log_probs = correct_log_probs[:, square_mask]
    return -selected_correct_log_probs.mean(dim=0).sum()


def count_linear_probe_parameters(d_model: int, num_squares: int | None = None) -> int:
    resolved_num_squares = BOARD_ROWS * BOARD_COLS if num_squares is None else num_squares
    return d_model * resolved_num_squares * BOARD_LABEL_OPTIONS


def count_module_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def make_probe_stem(layer: int, config: TPRProbeConfig) -> str:
    activation_tag = activation_name_to_tag(config.activation_name)
    stem = f"{activation_tag}_{layer}_tpr_r{config.role_dim}_f{config.filler_dim}"
    if config.use_bias:
        stem += "_bias"
    if config.exclude_center_squares:
        stem += "_no_center"
    return stem


def make_metrics_stem(config: TPRProbeConfig) -> str:
    activation_tag = activation_name_to_tag(config.activation_name)
    if activation_tag == "resid":
        stem = f"metrics_r{config.role_dim}_f{config.filler_dim}"
    else:
        stem = f"metrics_{activation_tag}_r{config.role_dim}_f{config.filler_dim}"
    if config.use_bias:
        stem += "_bias"
    if config.exclude_center_squares:
        stem += "_no_center"
    return stem


def make_probe_checkpoint_name(layer: int, config: TPRProbeConfig) -> str:
    return f"{make_probe_stem(layer, config)}_seed{config.seed}.pth"


def evaluate_tpr_probe(
    model,
    probe: TensorProductBoardProbe,
    layer: int,
    activation_name: str,
    games_tokens: Tensor,
    games_raw: Sequence[Sequence[int]],
    batch_size: int,
    pos_start: int,
    pos_end: int,
    device: torch.device,
    exclude_center_squares: bool = False,
) -> dict[str, float]:
    module_name = make_module_name(layer, activation_name)
    total_weight = 0
    total_loss = 0.0
    total_accuracy = 0.0
    square_num_correct = torch.zeros(BOARD_ROWS, BOARD_COLS, dtype=torch.float64)
    square_num_total = torch.zeros(BOARD_ROWS, BOARD_COLS, dtype=torch.float64)
    all_preds = []
    all_targets = []
    square_mask = build_square_selection_mask(
        exclude_center_squares=exclude_center_squares,
        device=device,
    )
    square_mask_flat = square_mask.reshape(-1)
    selected_square_count = int(square_mask.sum().item())

    probe.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(games_raw), batch_size):
            batch_end = min(batch_start + batch_size, len(games_raw))
            batch_raw = games_raw[batch_start:batch_end]
            batch_tokens = games_tokens[batch_start:batch_end].to(device)
            state_stack = build_state_stack(batch_raw)[:, pos_start:pos_end].to(device)
            state_stack_one_hot = state_stack_to_one_hot_threeway(state_stack)[0]

            with record_activations(model, [module_name]) as cache:
                model(batch_tokens[:, :-1])
            residual_activations = cache[module_name][0][:, pos_start:pos_end].to(device)
            probe_logits = compute_tpr_probe_logits(residual_activations, probe)
            loss = compute_tpr_probe_loss(
                probe_logits,
                state_stack_one_hot,
                square_mask=square_mask,
            )

            preds = probe_logits.argmax(dim=-1)
            targets = state_stack_one_hot.argmax(dim=-1)
            correct = (preds == targets).float()
            flat_correct = correct.reshape(correct.shape[0], correct.shape[1], -1)
            accuracy = (
                flat_correct[..., square_mask_flat].sum().item()
                / (correct.shape[0] * correct.shape[1] * selected_square_count)
            )
            square_num_correct += correct.sum(dim=(0, 1)).cpu().to(torch.float64)
            square_num_total += (
                square_mask.cpu().to(torch.float64) * (correct.shape[0] * correct.shape[1])
            )

            weight = batch_end - batch_start
            total_weight += weight
            total_loss += loss.item() * weight
            total_accuracy += accuracy * weight
            flat_preds = preds.reshape(preds.shape[0], preds.shape[1], -1)
            flat_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
            all_preds.append(flat_preds[..., square_mask_flat].reshape(-1).cpu())
            all_targets.append(flat_targets[..., square_mask_flat].reshape(-1).cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    square_accuracy = torch.full((BOARD_ROWS, BOARD_COLS), float("nan"), dtype=torch.float64)
    included_squares = square_num_total > 0
    square_accuracy[included_squares] = (
        square_num_correct[included_squares] / square_num_total[included_squares]
    )
    return {
        "loss": total_loss / total_weight,
        "accuracy": total_accuracy / total_weight,
        "macro_f1": macro_f1_score(preds, targets, num_classes=BOARD_LABEL_OPTIONS),
        "square_accuracy": square_accuracy.tolist(),
    }


def fit_binding_to_residual_linear_map(
    model,
    probe: TensorProductBoardProbe,
    layer: int,
    activation_name: str,
    games_tokens: Tensor,
    batch_size: int,
    pos_start: int,
    pos_end: int,
    device: torch.device,
    ridge: float,
) -> dict:
    if ridge < 0.0:
        raise ValueError("binding_to_residual_ridge must be non-negative")

    module_name = make_module_name(layer, activation_name)
    binding_feature_dim = probe.role_dim * probe.filler_dim
    augmented_feature_dim = binding_feature_dim + 1
    target_dim = probe.d_model

    gram = torch.zeros(
        augmented_feature_dim,
        augmented_feature_dim,
        dtype=torch.float64,
    )
    cross = torch.zeros(
        augmented_feature_dim,
        target_dim,
        dtype=torch.float64,
    )
    target_squared_sum = torch.zeros((), dtype=torch.float64)
    num_examples = 0

    probe.eval()
    binding_map = probe.binding_map.detach()
    with torch.inference_mode():
        for batch_start in range(0, games_tokens.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, games_tokens.shape[0])
            batch_tokens = games_tokens[batch_start:batch_end].to(device)

            with record_activations(model, [module_name]) as cache:
                model(batch_tokens[:, :-1])
            residual_activations = cache[module_name][0][:, pos_start:pos_end].to(device)

            binding_tensor = compute_tpr_binding_tensor(
                residual=residual_activations,
                binding_map=binding_map,
            )
            features = binding_tensor.reshape(-1, binding_feature_dim).to(
                dtype=torch.float64,
                device="cpu",
            )
            targets = residual_activations.reshape(-1, target_dim).to(
                dtype=torch.float64,
                device="cpu",
            )
            augmented_features = torch.cat(
                [
                    features,
                    torch.ones(features.shape[0], 1, dtype=torch.float64),
                ],
                dim=1,
            )

            gram += augmented_features.T @ augmented_features
            cross += augmented_features.T @ targets
            target_squared_sum += (targets * targets).sum()
            num_examples += int(augmented_features.shape[0])

    if num_examples == 0:
        raise ValueError("No training examples available to fit binding_to_residual")

    regularizer = torch.eye(augmented_feature_dim, dtype=torch.float64) * ridge
    regularizer[-1, -1] = 0.0
    solution = torch.linalg.lstsq(gram + regularizer, cross).solution
    weight = solution[:binding_feature_dim].to(dtype=torch.float32)
    bias = solution[binding_feature_dim].to(dtype=torch.float32)

    squared_error = target_squared_sum - 2.0 * (solution * cross).sum() + (
        solution * (gram @ solution)
    ).sum()
    squared_error = torch.clamp(squared_error, min=0.0)
    train_mse = float(
        squared_error / (num_examples * target_dim)
    )

    return {
        "weight": weight,
        "bias": bias,
        "input_dim": binding_feature_dim,
        "output_dim": target_dim,
        "fit_bias": True,
        "ridge": float(ridge),
        "num_examples": int(num_examples),
        "train_mse": train_mse,
    }


def load_saved_tpr_probe(
    probe_path: str | Path,
    d_model: int,
    device: torch.device,
) -> tuple[TensorProductBoardProbe, int, dict]:
    artifact = torch.load(probe_path, map_location=device)
    probe = TensorProductBoardProbe(
        d_model=d_model,
        role_dim=int(artifact["role_dim"]),
        filler_dim=int(artifact["filler_dim"]),
        use_bias=bool(artifact.get("use_bias", False)),
    ).to(device)
    probe.load_state_dict(artifact["probe_state_dict"])
    return probe, int(artifact["layer"]), artifact


def evaluate_saved_tpr_probe(
    model,
    probe_path: str | Path,
    games_tokens: Tensor,
    games_raw: Sequence[Sequence[int]],
    batch_size: int,
    pos_start: int,
    pos_end: int,
    device: torch.device,
    exclude_center_squares: bool = False,
) -> dict[str, float]:
    probe, layer, artifact = load_saved_tpr_probe(
        probe_path=probe_path,
        d_model=model.config.n_embd,
        device=device,
    )
    return evaluate_tpr_probe(
        model=model,
        probe=probe,
        layer=layer,
        activation_name=infer_activation_name_from_artifact(artifact),
        games_tokens=games_tokens,
        games_raw=games_raw,
        batch_size=batch_size,
        pos_start=pos_start,
        pos_end=pos_end,
        device=device,
        exclude_center_squares=exclude_center_squares,
    )


def train(config: TPRProbeConfig) -> dict[int, dict[str, float]]:
    config.activation_name = normalize_activation_name(config.activation_name)
    if config.binding_to_residual_ridge < 0.0:
        raise ValueError("--binding-to-residual-ridge must be non-negative")
    print("TPR probe config:")
    print(json.dumps(asdict(config), indent=2))
    seed_all(config.seed)

    device = torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(
        {
            "model_path": config.checkpoint,
            "device": device,
            "load_weights": not config.random_init,
            "n_head": config.n_head,
        }
    )
    if config.random_init:
        print(
            f"Using randomly initialized transformer with architecture inferred from {config.checkpoint}"
        )
    convert_to_hooked_model(model)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_probe_dataset(
        data_path=config.data_path,
        block_size=model.get_block_size(),
        valid_size=config.valid_size,
        test_size=config.test_size,
        seed=config.seed,
        max_games=config.max_games,
    )
    print(
        "Dataset sizes:",
        json.dumps(
            {
                "train": len(split["train_raw"]),
                "valid": len(split["valid_raw"]),
                "test": len(split["test_raw"]),
            },
            indent=2,
        ),
    )

    pos_start, pos_end = resolve_position_slice(
        config.pos_start,
        config.pos_end,
        model.get_block_size(),
    )
    square_mask = build_square_selection_mask(
        exclude_center_squares=config.exclude_center_squares,
        device=device,
    )
    layers = select_layers(config, model)
    train_tokens = split["train_tokens"]
    train_raw = split["train_raw"]
    linear_probe_parameter_count = count_linear_probe_parameters(
        model.config.n_embd,
        num_squares=int(square_mask.sum().item()),
    )

    results = {}
    for layer in layers:
        probe_path = output_dir / make_probe_checkpoint_name(layer, config)
        module_name = make_module_name(layer, config.activation_name)
        print(f"Training layer {layer}")

        probe = TensorProductBoardProbe(
            d_model=model.config.n_embd,
            role_dim=config.role_dim,
            filler_dim=config.filler_dim,
            use_bias=config.use_bias,
        ).to(device)
        probe_parameter_count = count_module_parameters(probe)
        print(
            "  Parameter counts:",
            json.dumps(
                {
                    "tpr_probe": probe_parameter_count,
                    "linear_probe": linear_probe_parameter_count,
                    "tpr_vs_linear_ratio": probe_parameter_count
                    / linear_probe_parameter_count,
                },
                indent=2,
            ),
        )
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.wd,
        )

        best_valid_loss = float("inf")
        patience = 0
        seen = 0
        stop = False

        for epoch in range(config.num_epochs):
            if stop:
                break

            epoch_indices = torch.randperm(len(train_raw))
            progress = tqdm(
                range(0, len(train_raw), config.batch_size),
                desc=f"layer {layer} epoch {epoch}",
            )
            probe.train()
            for batch_start in progress:
                batch_end = min(batch_start + config.batch_size, len(train_raw))
                batch_indices = epoch_indices[batch_start:batch_end].tolist()
                batch_tokens = train_tokens[batch_indices].to(device)
                batch_raw = [train_raw[idx] for idx in batch_indices]

                state_stack = build_state_stack(batch_raw)[:, pos_start:pos_end].to(
                    device
                )
                state_stack_one_hot = state_stack_to_one_hot_threeway(state_stack)[0]

                with torch.no_grad():
                    with record_activations(model, [module_name]) as cache:
                        model(batch_tokens[:, :-1])
                residual_activations = (
                    cache[module_name][0][:, pos_start:pos_end].to(device).clone()
                )

                optimizer.zero_grad(set_to_none=True)
                probe_logits = compute_tpr_probe_logits(residual_activations, probe)
                loss = compute_tpr_probe_loss(
                    probe_logits,
                    state_stack_one_hot,
                    square_mask=square_mask,
                )
                loss.backward()
                optimizer.step()

                seen += batch_end - batch_start
                progress.set_postfix(loss=float(loss.item()))

                batch_id = batch_start // config.batch_size
                if batch_id % config.valid_every == 0:
                    metrics = evaluate_tpr_probe(
                        model=model,
                        probe=probe,
                        layer=layer,
                        activation_name=config.activation_name,
                        games_tokens=split["valid_tokens"],
                        games_raw=split["valid_raw"],
                        batch_size=config.batch_size,
                        pos_start=pos_start,
                        pos_end=pos_end,
                        device=device,
                        exclude_center_squares=config.exclude_center_squares,
                    )
                    print(
                        f"  Validation loss={metrics['loss']:.4f} "
                        f"accuracy={metrics['accuracy']:.4f} "
                        f"macro_f1={metrics['macro_f1']:.4f}"
                    )
                    print("  Validation accuracy by square:")
                    print(
                        format_square_accuracy_board(
                            torch.tensor(metrics["square_accuracy"])
                        )
                    )

                    if metrics["loss"] < best_valid_loss:
                        best_valid_loss = metrics["loss"]
                        patience = 0
                        torch.save(
                            {
                                "probe_kind": "tensor_product",
                                "probe_state_dict": probe.state_dict(),
                                "layer": layer,
                                "metrics": metrics,
                                "config": asdict(config),
                                "module_name": module_name,
                                "activation_name": config.activation_name,
                                "role_dim": config.role_dim,
                                "filler_dim": config.filler_dim,
                                "use_bias": config.use_bias,
                                "probe_parameter_count": probe_parameter_count,
                                "linear_probe_parameter_count": linear_probe_parameter_count,
                                "parameter_ratio": probe_parameter_count
                                / linear_probe_parameter_count,
                            },
                            probe_path,
                        )
                    else:
                        patience += 1
                        print(f"  Patience {patience}/{config.valid_patience}")
                        if patience >= config.valid_patience:
                            print(
                                f"  Early stopping layer {layer} after seeing {seen} games"
                            )
                            stop = True
                            break

        saved_probe, saved_layer, saved_artifact = load_saved_tpr_probe(
            probe_path=probe_path,
            d_model=model.config.n_embd,
            device=device,
        )
        if saved_layer != layer:
            raise ValueError(
                f"Loaded saved TPR probe layer {saved_layer} from {probe_path}, "
                f"expected layer {layer}"
            )

        binding_to_residual = fit_binding_to_residual_linear_map(
            model=model,
            probe=saved_probe,
            layer=saved_layer,
            activation_name=infer_activation_name_from_artifact(saved_artifact),
            games_tokens=train_tokens,
            batch_size=config.batch_size,
            pos_start=pos_start,
            pos_end=pos_end,
            device=device,
            ridge=config.binding_to_residual_ridge,
        )
        saved_artifact["binding_to_residual"] = binding_to_residual
        torch.save(saved_artifact, probe_path)
        print(
            "  Fitted binding->residual linear map "
            f"(train_mse={binding_to_residual['train_mse']:.6f}, "
            f"samples={binding_to_residual['num_examples']})"
        )

        test_metrics = evaluate_saved_tpr_probe(
            model=model,
            probe_path=probe_path,
            games_tokens=split["test_tokens"],
            games_raw=split["test_raw"],
            batch_size=config.batch_size,
            pos_start=pos_start,
            pos_end=pos_end,
            device=device,
            exclude_center_squares=config.exclude_center_squares,
        )
        print(
            f"  Test loss={test_metrics['loss']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        results[layer] = {
            **test_metrics,
            "probe_parameter_count": probe_parameter_count,
            "linear_probe_parameter_count": linear_probe_parameter_count,
            "parameter_ratio": probe_parameter_count / linear_probe_parameter_count,
            "binding_to_residual_train_mse": binding_to_residual["train_mse"],
            "binding_to_residual_num_examples": binding_to_residual["num_examples"],
            "binding_to_residual_ridge": binding_to_residual["ridge"],
        }

    metrics_name = make_metrics_stem(config)
    with (output_dir / f"{metrics_name}.json").open("w", encoding="utf-8") as file_p:
        json.dump(results, file_p, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=TPRProbeConfig.checkpoint)
    parser.add_argument(
        "--data-path",
        default=TPRProbeConfig.data_path,
        help="Path to a pickle dataset file or directory of pickle shards.",
    )
    parser.add_argument("--output-dir", default=TPRProbeConfig.output_dir)
    parser.add_argument("--device", default=TPRProbeConfig.device)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help=(
            "Use a randomly initialized transformer with architecture inferred "
            "from --checkpoint instead of loading pretrained weights."
        ),
    )
    parser.add_argument("--n-head", type=int, default=TPRProbeConfig.n_head)
    parser.add_argument("--lr", type=float, default=TPRProbeConfig.lr)
    parser.add_argument("--wd", type=float, default=TPRProbeConfig.wd)
    parser.add_argument("--batch-size", type=int, default=TPRProbeConfig.batch_size)
    parser.add_argument("--valid-every", type=int, default=TPRProbeConfig.valid_every)
    parser.add_argument("--pos-start", type=int, default=TPRProbeConfig.pos_start)
    parser.add_argument(
        "--pos-end",
        type=int,
        default=TPRProbeConfig.pos_end,
        help="Exclusive end position for the probed move slice; defaults to block_size.",
    )
    parser.add_argument("--num-epochs", type=int, default=TPRProbeConfig.num_epochs)
    parser.add_argument("--valid-size", type=int, default=TPRProbeConfig.valid_size)
    parser.add_argument("--test-size", type=int, default=TPRProbeConfig.test_size)
    parser.add_argument(
        "--valid-patience", type=int, default=TPRProbeConfig.valid_patience
    )
    parser.add_argument("--seed", type=int, default=TPRProbeConfig.seed)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument(
        "--train-layers",
        type=str,
        default=TPRProbeConfig.train_layers,
        help="Comma-separated residual layers to probe.",
    )
    parser.add_argument(
        "--role-dim",
        type=int,
        default=TPRProbeConfig.role_dim,
        help="Dimensionality of learned square-role vectors.",
    )
    parser.add_argument(
        "--filler-dim",
        type=int,
        default=TPRProbeConfig.filler_dim,
        help="Dimensionality of learned board-content filler vectors.",
    )
    parser.add_argument(
        "--use-bias",
        action="store_true",
        help="Add an unconstrained per-square, per-class bias term.",
    )
    parser.add_argument(
        "--exclude-center-squares",
        action="store_true",
        help=(
            "Exclude the four starting center squares from the TPR probe loss and "
            "reported metrics."
        ),
    )
    parser.add_argument(
        "--activation-name",
        type=normalize_activation_name,
        default=TPRProbeConfig.activation_name,
        help=(
            "Residual hook to train against. Accepted aliases: resid_post, "
            "hook_resid_post, resid_mid, hook_resid_mid."
        ),
    )
    parser.add_argument(
        "--binding-to-residual-ridge",
        type=float,
        default=TPRProbeConfig.binding_to_residual_ridge,
        help=(
            "Ridge regularization used when fitting the post-training linear map "
            "from flattened binding tensors back to residual activations."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TPRProbeConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        random_init=args.random_init,
        n_head=args.n_head,
        lr=args.lr,
        wd=args.wd,
        batch_size=args.batch_size,
        valid_every=args.valid_every,
        pos_start=args.pos_start,
        pos_end=args.pos_end,
        num_epochs=args.num_epochs,
        valid_size=args.valid_size,
        test_size=args.test_size,
        valid_patience=args.valid_patience,
        seed=args.seed,
        max_games=args.max_games,
        max_layers=args.max_layers,
        train_layers=args.train_layers,
        role_dim=args.role_dim,
        filler_dim=args.filler_dim,
        use_bias=args.use_bias,
        exclude_center_squares=args.exclude_center_squares,
        activation_name=args.activation_name,
        binding_to_residual_ridge=args.binding_to_residual_ridge,
    )
    train(config)


if __name__ == "__main__":
    main()
