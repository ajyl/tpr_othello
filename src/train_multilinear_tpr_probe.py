"""Train multilinear tensor-product board probes on Othello GPT activations."""

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
from train_tpr_probe import (  # noqa: E402
    DEFAULT_BINDING_TO_RESIDUAL_RIDGE,
    DEFAULT_TPR_ACTIVATION_NAME,
    activation_name_to_tag,
    build_square_selection_mask,
    compute_tpr_probe_loss,
    count_linear_probe_parameters,
    count_module_parameters,
    infer_activation_name_from_artifact,
    make_module_name,
    normalize_activation_name,
)


class MultilinearTensorProductBoardProbe(torch.nn.Module):
    """
    Probe that decodes board state through a row x column x color binding tensor.

    For each residual vector h_t, the probe learns a linear map into a 3-way
    binding tensor G_t with axes for rows, columns, and relative board labels:

        G_t = sum_d h_t[d] * U[d]
        logit[i, j, c] = sum_{a,b,k} row_i[a] col_j[b] color_c[k] G_t[a, b, k]

    This constrains square/class logits to arise from separate row, column, and
    color factors rather than a free square-role matrix.
    """

    def __init__(
        self,
        d_model: int,
        row_dim: int,
        col_dim: int,
        color_dim: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if row_dim <= 0:
            raise ValueError("row_dim must be positive")
        if col_dim <= 0:
            raise ValueError("col_dim must be positive")
        if color_dim <= 0:
            raise ValueError("color_dim must be positive")

        self.d_model = int(d_model)
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)
        self.color_dim = int(color_dim)
        self.use_bias = bool(use_bias)

        self.binding_map = torch.nn.Parameter(
            torch.randn(d_model, row_dim, col_dim, color_dim) / math.sqrt(d_model)
        )
        self.row_embeddings = torch.nn.Parameter(
            torch.randn(BOARD_ROWS, row_dim) / math.sqrt(row_dim)
        )
        self.col_embeddings = torch.nn.Parameter(
            torch.randn(BOARD_COLS, col_dim) / math.sqrt(col_dim)
        )
        self.color_embeddings = torch.nn.Parameter(
            torch.randn(BOARD_LABEL_OPTIONS, color_dim) / math.sqrt(color_dim)
        )
        if self.use_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(BOARD_ROWS, BOARD_COLS, BOARD_LABEL_OPTIONS)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, residual: Tensor) -> Tensor:
        binding_tensor = torch.einsum(
            "bpd,drck->bprck",
            residual,
            self.binding_map,
        )
        logits = torch.einsum(
            "bprck,ir,jc,ok->bpijo",
            binding_tensor,
            self.row_embeddings,
            self.col_embeddings,
            self.color_embeddings,
        )
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def factor_parameter_count(self) -> int:
        return (
            self.row_embeddings.numel()
            + self.col_embeddings.numel()
            + self.color_embeddings.numel()
        )

    def binding_feature_dim(self) -> int:
        return self.row_dim * self.col_dim * self.color_dim


@dataclass
class MultilinearTPRProbeConfig:
    checkpoint: str = "ckpts/synthetic_model.pth"
    data_path: str = "train_data"
    output_dir: str = "probes/tpr_multilinear"
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
    row_dim: int = 8
    col_dim: int = 8
    color_dim: int = 4
    use_bias: bool = False
    exclude_center_squares: bool = False
    activation_name: str = DEFAULT_TPR_ACTIVATION_NAME
    binding_to_residual_ridge: float = DEFAULT_BINDING_TO_RESIDUAL_RIDGE


def compute_multilinear_binding_tensor(
    residual: Tensor,
    binding_map: Tensor,
) -> Tensor:
    return torch.einsum("bpd,drck->bprck", residual, binding_map)


def count_multilinear_probe_parameters(
    d_model: int,
    row_dim: int,
    col_dim: int,
    color_dim: int,
    use_bias: bool,
) -> int:
    total = d_model * row_dim * col_dim * color_dim
    total += BOARD_ROWS * row_dim
    total += BOARD_COLS * col_dim
    total += BOARD_LABEL_OPTIONS * color_dim
    if use_bias:
        total += BOARD_ROWS * BOARD_COLS * BOARD_LABEL_OPTIONS
    return total


def make_probe_stem(layer: int, config: MultilinearTPRProbeConfig) -> str:
    activation_tag = activation_name_to_tag(config.activation_name)
    stem = (
        f"{activation_tag}_{layer}_mltpr"
        f"_row{config.row_dim}_col{config.col_dim}_color{config.color_dim}"
    )
    if config.use_bias:
        stem += "_bias"
    if config.exclude_center_squares:
        stem += "_no_center"
    return stem


def make_metrics_stem(config: MultilinearTPRProbeConfig) -> str:
    activation_tag = activation_name_to_tag(config.activation_name)
    if activation_tag == "resid":
        stem = (
            f"metrics_mltpr_row{config.row_dim}"
            f"_col{config.col_dim}_color{config.color_dim}"
        )
    else:
        stem = (
            f"metrics_{activation_tag}_mltpr_row{config.row_dim}"
            f"_col{config.col_dim}_color{config.color_dim}"
        )
    if config.use_bias:
        stem += "_bias"
    if config.exclude_center_squares:
        stem += "_no_center"
    return stem


def make_probe_checkpoint_name(layer: int, config: MultilinearTPRProbeConfig) -> str:
    return f"{make_probe_stem(layer, config)}_seed{config.seed}.pth"


def evaluate_multilinear_tpr_probe(
    model,
    probe: MultilinearTensorProductBoardProbe,
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
            probe_logits = probe(residual_activations)
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


def fit_multilinear_binding_to_residual_linear_map(
    model,
    probe: MultilinearTensorProductBoardProbe,
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
    binding_feature_dim = probe.binding_feature_dim()
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

            binding_tensor = compute_multilinear_binding_tensor(
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
    train_mse = float(squared_error / (num_examples * target_dim))

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


def load_saved_multilinear_tpr_probe(
    probe_path: str | Path,
    d_model: int,
    device: torch.device,
) -> tuple[MultilinearTensorProductBoardProbe, int, dict]:
    artifact = torch.load(probe_path, map_location=device)
    probe = MultilinearTensorProductBoardProbe(
        d_model=d_model,
        row_dim=int(artifact["row_dim"]),
        col_dim=int(artifact["col_dim"]),
        color_dim=int(artifact["color_dim"]),
        use_bias=bool(artifact.get("use_bias", False)),
    ).to(device)
    probe.load_state_dict(artifact["probe_state_dict"])
    return probe, int(artifact["layer"]), artifact


def evaluate_saved_multilinear_tpr_probe(
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
    probe, layer, artifact = load_saved_multilinear_tpr_probe(
        probe_path=probe_path,
        d_model=model.config.n_embd,
        device=device,
    )
    return evaluate_multilinear_tpr_probe(
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


def train(config: MultilinearTPRProbeConfig) -> dict[int, dict[str, float]]:
    config.activation_name = normalize_activation_name(config.activation_name)
    if config.row_dim <= 0:
        raise ValueError("--row-dim must be positive")
    if config.col_dim <= 0:
        raise ValueError("--col-dim must be positive")
    if config.color_dim <= 0:
        raise ValueError("--color-dim must be positive")
    if config.binding_to_residual_ridge < 0.0:
        raise ValueError("--binding-to-residual-ridge must be non-negative")

    print("Multilinear TPR probe config:")
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
    expected_multilinear_parameter_count = count_multilinear_probe_parameters(
        d_model=model.config.n_embd,
        row_dim=config.row_dim,
        col_dim=config.col_dim,
        color_dim=config.color_dim,
        use_bias=config.use_bias,
    )

    results = {}
    for layer in layers:
        probe_path = output_dir / make_probe_checkpoint_name(layer, config)
        module_name = make_module_name(layer, config.activation_name)
        print(f"Training layer {layer}")

        probe = MultilinearTensorProductBoardProbe(
            d_model=model.config.n_embd,
            row_dim=config.row_dim,
            col_dim=config.col_dim,
            color_dim=config.color_dim,
            use_bias=config.use_bias,
        ).to(device)
        probe_parameter_count = count_module_parameters(probe)
        factor_parameter_count = probe.factor_parameter_count()
        if probe_parameter_count != expected_multilinear_parameter_count:
            raise ValueError(
                "Multilinear parameter count mismatch: "
                f"{probe_parameter_count} != {expected_multilinear_parameter_count}"
            )

        print(
            "  Parameter counts:",
            json.dumps(
                {
                    "multilinear_probe": probe_parameter_count,
                    "linear_probe": linear_probe_parameter_count,
                    "multilinear_vs_linear_ratio": probe_parameter_count
                    / linear_probe_parameter_count,
                    "factor_parameters": factor_parameter_count,
                    "binding_parameters": probe.binding_map.numel(),
                    "binding_feature_dim": probe.binding_feature_dim(),
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
                probe_logits = probe(residual_activations)
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
                    metrics = evaluate_multilinear_tpr_probe(
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
                                "probe_kind": "multilinear_tensor_product",
                                "probe_state_dict": probe.state_dict(),
                                "layer": layer,
                                "metrics": metrics,
                                "config": asdict(config),
                                "module_name": module_name,
                                "activation_name": config.activation_name,
                                "row_dim": config.row_dim,
                                "col_dim": config.col_dim,
                                "color_dim": config.color_dim,
                                "use_bias": config.use_bias,
                                "probe_parameter_count": probe_parameter_count,
                                "linear_probe_parameter_count": linear_probe_parameter_count,
                                "parameter_ratio": probe_parameter_count
                                / linear_probe_parameter_count,
                                "factor_parameter_count": factor_parameter_count,
                                "binding_parameter_count": probe.binding_map.numel(),
                                "binding_feature_dim": probe.binding_feature_dim(),
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

        saved_probe, saved_layer, saved_artifact = load_saved_multilinear_tpr_probe(
            probe_path=probe_path,
            d_model=model.config.n_embd,
            device=device,
        )
        if saved_layer != layer:
            raise ValueError(
                f"Loaded saved multilinear probe layer {saved_layer} from {probe_path}, "
                f"expected layer {layer}"
            )

        binding_to_residual = fit_multilinear_binding_to_residual_linear_map(
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

        test_metrics = evaluate_saved_multilinear_tpr_probe(
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
            "factor_parameter_count": factor_parameter_count,
            "binding_parameter_count": probe.binding_map.numel(),
            "binding_feature_dim": probe.binding_feature_dim(),
            "binding_to_residual_train_mse": binding_to_residual["train_mse"],
            "binding_to_residual_num_examples": binding_to_residual["num_examples"],
            "binding_to_residual_ridge": binding_to_residual["ridge"],
            "row_dim": config.row_dim,
            "col_dim": config.col_dim,
            "color_dim": config.color_dim,
        }

    metrics_name = make_metrics_stem(config)
    with (output_dir / f"{metrics_name}.json").open("w", encoding="utf-8") as file_p:
        json.dump(results, file_p, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=MultilinearTPRProbeConfig.checkpoint)
    parser.add_argument(
        "--data-path",
        default=MultilinearTPRProbeConfig.data_path,
        help="Path to a pickle dataset file or directory of pickle shards.",
    )
    parser.add_argument("--output-dir", default=MultilinearTPRProbeConfig.output_dir)
    parser.add_argument("--device", default=MultilinearTPRProbeConfig.device)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help=(
            "Use a randomly initialized transformer with architecture inferred "
            "from --checkpoint instead of loading pretrained weights."
        ),
    )
    parser.add_argument("--n-head", type=int, default=MultilinearTPRProbeConfig.n_head)
    parser.add_argument("--lr", type=float, default=MultilinearTPRProbeConfig.lr)
    parser.add_argument("--wd", type=float, default=MultilinearTPRProbeConfig.wd)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MultilinearTPRProbeConfig.batch_size,
    )
    parser.add_argument(
        "--valid-every",
        type=int,
        default=MultilinearTPRProbeConfig.valid_every,
    )
    parser.add_argument(
        "--pos-start",
        type=int,
        default=MultilinearTPRProbeConfig.pos_start,
    )
    parser.add_argument(
        "--pos-end",
        type=int,
        default=MultilinearTPRProbeConfig.pos_end,
        help="Exclusive end position for the probed move slice; defaults to block_size.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=MultilinearTPRProbeConfig.num_epochs,
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=MultilinearTPRProbeConfig.valid_size,
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=MultilinearTPRProbeConfig.test_size,
    )
    parser.add_argument(
        "--valid-patience",
        type=int,
        default=MultilinearTPRProbeConfig.valid_patience,
    )
    parser.add_argument("--seed", type=int, default=MultilinearTPRProbeConfig.seed)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument(
        "--train-layers",
        type=str,
        default=MultilinearTPRProbeConfig.train_layers,
        help="Comma-separated residual layers to probe.",
    )
    parser.add_argument(
        "--row-dim",
        type=int,
        default=MultilinearTPRProbeConfig.row_dim,
        help="Dimensionality of the learned row embeddings.",
    )
    parser.add_argument(
        "--col-dim",
        type=int,
        default=MultilinearTPRProbeConfig.col_dim,
        help="Dimensionality of the learned column embeddings.",
    )
    parser.add_argument(
        "--color-dim",
        type=int,
        default=MultilinearTPRProbeConfig.color_dim,
        help=(
            "Dimensionality of the learned color embeddings. "
            "These correspond to the three relative board labels."
        ),
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
            "Exclude the four starting center squares from the training loss and "
            "reported metrics."
        ),
    )
    parser.add_argument(
        "--activation-name",
        type=normalize_activation_name,
        default=MultilinearTPRProbeConfig.activation_name,
        help=(
            "Residual hook to train against. Accepted aliases: resid_post, "
            "hook_resid_post, resid_mid, hook_resid_mid."
        ),
    )
    parser.add_argument(
        "--binding-to-residual-ridge",
        type=float,
        default=MultilinearTPRProbeConfig.binding_to_residual_ridge,
        help=(
            "Ridge regularization used when fitting the post-training linear map "
            "from flattened multilinear binding tensors back to residual activations."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MultilinearTPRProbeConfig(
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
        row_dim=args.row_dim,
        col_dim=args.col_dim,
        color_dim=args.color_dim,
        use_bias=args.use_bias,
        exclude_center_squares=args.exclude_center_squares,
        activation_name=args.activation_name,
        binding_to_residual_ridge=args.binding_to_residual_ridge,
    )
    train(config)


if __name__ == "__main__":
    main()
