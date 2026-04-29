"""Evaluate rank-k truncated SVD baselines for a saved linear board-state probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "hook_utils"))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from hook_utils import (
    convert_to_hooked_model,
    record_activations,
    seed_all,
)  # noqa: E402
from load_model import load_model  # noqa: E402
from train_probe import (  # noqa: E402
    BOARD_COLS,
    BOARD_LABEL_OPTIONS,
    BOARD_ROWS,
    ProbeConfig,
    build_state_stack,
    format_square_accuracy_board,
    load_probe_dataset,
    resolve_position_slice,
    state_stack_to_one_hot_threeway,
)
from train_tpr_probe import (  # noqa: E402
    TPRProbeConfig,
    TensorProductBoardProbe,
    infer_activation_name_from_artifact,
    load_saved_tpr_probe,
    make_module_name,
)


NUM_BOARD_OUTPUTS = BOARD_ROWS * BOARD_COLS * BOARD_LABEL_OPTIONS


@dataclass
class LinearProbeSVDBaselineConfig:
    linear_probe_path: str
    tpr_probe_path: str
    checkpoint: str | None = None
    data_path: str | None = None
    output_path: str | None = None
    plot_output_path: str | None = None
    device: str = "auto"
    n_head: int | None = None
    batch_size: int | None = None
    split: str = "test"
    valid_size: int | None = None
    test_size: int | None = None
    seed: int | None = None
    max_games: int | None = None
    pos_start: int | None = None
    pos_end: int | None = None
    num_auto_ranks: int = 17
    num_plot_points: int | None = None
    x_axis_units: str = "ratio"
    ranks: tuple[int, ...] = ()
    skip_tpr_eval: bool = False
    use_effective_tpr_weights: bool = False


@dataclass
class EvaluationContext:
    checkpoint: Path
    data_path: Path
    batch_size: int
    valid_size: int
    test_size: int
    seed: int
    max_games: int | None
    pos_start: int
    pos_end: int
    primary_square_count: int
    split: str
    split_tokens_key: str
    split_raw_key: str


@dataclass
class LoadedTPRLikeProbe:
    probe: torch.nn.Module
    artifact: dict
    probe_kind: str
    is_baseline: bool
    layer: int | None
    module_name: str | None
    square_color_vectors: Tensor | None


class MetricAccumulator:
    def __init__(self) -> None:
        self.total_weight = 0
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.square_num_correct = torch.zeros(
            BOARD_ROWS, BOARD_COLS, dtype=torch.float64
        )
        self.square_num_total = torch.zeros(BOARD_ROWS, BOARD_COLS, dtype=torch.float64)
        self.confusion = torch.zeros(
            BOARD_LABEL_OPTIONS,
            BOARD_LABEL_OPTIONS,
            dtype=torch.int64,
        )

    def update(
        self,
        logits: Tensor,
        targets_one_hot: Tensor,
        square_mask: Tensor,
    ) -> None:
        square_mask = square_mask.to(device=logits.device, dtype=torch.bool)
        square_mask_flat = square_mask.reshape(-1)
        selected_square_count = int(square_mask.sum().item())
        if selected_square_count <= 0:
            raise ValueError("square_mask must select at least one square")

        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = (log_probs * targets_one_hot).sum(dim=-1).mean(dim=0)
        correct_log_probs = correct_log_probs * BOARD_LABEL_OPTIONS
        loss = -correct_log_probs[:, square_mask].mean(dim=0).sum()

        preds = logits.argmax(dim=-1)
        targets = targets_one_hot.argmax(dim=-1)
        correct = (preds == targets).float()
        flat_correct = correct.reshape(correct.shape[0], correct.shape[1], -1)
        accuracy = flat_correct[..., square_mask_flat].sum().item() / (
            correct.shape[0] * correct.shape[1] * selected_square_count
        )

        selected_preds = preds.reshape(preds.shape[0], preds.shape[1], -1)[
            ..., square_mask_flat
        ].reshape(-1)
        selected_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)[
            ..., square_mask_flat
        ].reshape(-1)
        confusion_update = torch.bincount(
            (selected_targets * BOARD_LABEL_OPTIONS + selected_preds).cpu(),
            minlength=BOARD_LABEL_OPTIONS * BOARD_LABEL_OPTIONS,
        ).reshape(BOARD_LABEL_OPTIONS, BOARD_LABEL_OPTIONS)

        batch_weight = int(logits.shape[0])
        self.total_weight += batch_weight
        self.total_loss += float(loss.item()) * batch_weight
        self.total_accuracy += accuracy * batch_weight
        self.square_num_correct += correct.sum(dim=(0, 1)).cpu().to(torch.float64)
        self.square_num_total += square_mask.cpu().to(torch.float64) * (
            correct.shape[0] * correct.shape[1]
        )
        self.confusion += confusion_update.to(torch.int64)

    def finalize(self) -> dict[str, float | list[list[float]]]:
        if self.total_weight <= 0:
            raise ValueError("No examples were accumulated")

        f1_scores = []
        for class_id in range(BOARD_LABEL_OPTIONS):
            true_positive = float(self.confusion[class_id, class_id].item())
            false_positive = float(
                self.confusion[:, class_id].sum().item() - true_positive
            )
            false_negative = float(
                self.confusion[class_id, :].sum().item() - true_positive
            )
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive > 0.0
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative > 0.0
                else 0.0
            )
            if precision + recall == 0.0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2.0 * precision * recall / (precision + recall))
        macro_f1 = float(sum(f1_scores) / len(f1_scores))

        square_accuracy = torch.full(
            (BOARD_ROWS, BOARD_COLS), float("nan"), dtype=torch.float64
        )
        included_squares = self.square_num_total > 0
        square_accuracy[included_squares] = (
            self.square_num_correct[included_squares]
            / self.square_num_total[included_squares]
        )
        return {
            "loss": self.total_loss / self.total_weight,
            "accuracy": self.total_accuracy / self.total_weight,
            "macro_f1": macro_f1,
            "square_accuracy": square_accuracy.tolist(),
        }



def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()
    root_relative = (ROOT / path).resolve()
    if root_relative.exists():
        return root_relative
    return path


def choose_config_value(
    cli_value,
    preferred_config: dict,
    fallback_config: dict,
    key: str,
    default=None,
):
    if cli_value is not None:
        return cli_value
    if key in preferred_config and preferred_config[key] is not None:
        return preferred_config[key]
    if key in fallback_config and fallback_config[key] is not None:
        return fallback_config[key]
    return default


def normalize_pos_end(raw_pos_end) -> int | None:
    if raw_pos_end is None:
        return None
    resolved = int(raw_pos_end)
    if resolved <= 0:
        return None
    return resolved


def load_linear_probe_artifact(
    probe_path: Path,
    device: torch.device,
) -> tuple[Tensor, int, str, dict]:
    artifact = torch.load(probe_path, map_location=device)
    if not isinstance(artifact, dict):
        raise TypeError(
            f"Expected a dict checkpoint for {probe_path}, got {type(artifact)}"
        )
    if "probe" not in artifact:
        raise KeyError(f"Missing `probe` tensor in {probe_path}")
    probe = artifact["probe"]
    if not isinstance(probe, torch.Tensor):
        raise TypeError(f"Expected `probe` to be a tensor in {probe_path}")
    if probe.ndim != 5 or probe.shape[0] != 1:
        raise ValueError(
            "Expected a linear probe tensor of shape [1, d_model, 8, 8, 3], "
            f"got {tuple(probe.shape)}"
        )
    layer = artifact.get("layer")
    if layer is None:
        raise KeyError(f"Missing `layer` in {probe_path}")
    module_name = artifact.get("module_name")
    if not isinstance(module_name, str) or not module_name:
        module_name = f"blocks.{int(layer)}.hook_resid_post"
    return probe.to(device), int(layer), module_name, artifact


def load_tpr_like_probe(
    probe_path: Path,
    *,
    linear_d_model: int,
    device: torch.device,
) -> LoadedTPRLikeProbe:
    artifact = torch.load(probe_path, map_location=device)
    if not isinstance(artifact, dict):
        raise TypeError(
            f"Expected a dict checkpoint for {probe_path}, got {type(artifact)}"
        )

    probe_kind = str(artifact.get("probe_kind", "tensor_product"))
    probe, layer, artifact = load_saved_tpr_probe(
        probe_path=probe_path,
        d_model=linear_d_model,
        device=device,
    )
    activation_name = infer_activation_name_from_artifact(artifact)
    module_name = make_module_name(layer, activation_name)
    return LoadedTPRLikeProbe(
        probe=probe,
        artifact=artifact,
        probe_kind=probe_kind,
        is_baseline=False,
        layer=layer,
        module_name=module_name,
        square_color_vectors=None,
    )


def flatten_linear_probe(probe: Tensor) -> Tensor:
    flattened = probe.squeeze(0).reshape(probe.shape[1], -1)
    if flattened.shape[1] != NUM_BOARD_OUTPUTS:
        raise ValueError(
            "Expected the flattened linear probe to have 192 outputs, "
            f"got shape {tuple(flattened.shape)}"
        )
    return flattened


def flatten_effective_weight_tensor(weight_tensor: Tensor) -> Tensor:
    if weight_tensor.ndim != 4:
        raise ValueError(
            "Expected an effective weight tensor of shape [d_model, 8, 8, 3], "
            f"got {tuple(weight_tensor.shape)}"
        )
    flattened = weight_tensor.reshape(weight_tensor.shape[0], -1)
    if flattened.shape[1] != NUM_BOARD_OUTPUTS:
        raise ValueError(
            "Expected the flattened effective weight tensor to have 192 outputs, "
            f"got shape {tuple(flattened.shape)}"
        )
    return flattened


def flatten_output_bias(bias: Tensor) -> Tensor:
    flattened = bias.reshape(-1)
    if flattened.shape[0] != NUM_BOARD_OUTPUTS:
        raise ValueError(
            "Expected the flattened output bias to have 192 entries, "
            f"got shape {tuple(flattened.shape)}"
        )
    return flattened


def compute_tpr_effective_weights(
    probe: torch.nn.Module,
) -> tuple[Tensor, Tensor | None]:
    if isinstance(probe, TensorProductBoardProbe):
        effective_weights = torch.einsum(
            "drf,xyr,cf->dxyc",
            probe.binding_map,
            probe.role_embeddings,
            probe.filler_embeddings,
        )
    else:
        raise TypeError(
            "Unsupported probe type for effective-weight computation: " f"{type(probe)}"
        )
    bias = probe.bias
    return effective_weights, bias


def compute_rank_parameter_count(
    rank: int,
    d_model: int,
    num_outputs: int,
    *,
    bias_parameter_count: int = 0,
) -> int:
    return int(rank) * (int(d_model) + int(num_outputs)) + int(bias_parameter_count)


def compute_truncated_svd_factors(
    weight_matrix: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if weight_matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D weight matrix for SVD, got shape {tuple(weight_matrix.shape)}"
        )
    output_dim, d_model = weight_matrix.shape
    max_rank = min(output_dim, d_model)
    if max_rank <= 0:
        raise ValueError("Cannot factorize an empty weight matrix")

    u, singular_values, vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    encoder = vh.T.contiguous()
    decoder = (u * singular_values.unsqueeze(0)).T.contiguous()
    return encoder, decoder, singular_values


def build_auto_rank_sweep(
    *,
    max_rank: int,
    target_rank: float,
    num_auto_ranks: int,
) -> list[int]:
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")

    ranks: set[int] = {1, max_rank}

    log_steps = max(num_auto_ranks, 2)
    for step_idx in range(log_steps):
        frac = step_idx / max(log_steps - 1, 1)
        rank = int(round(math.exp(frac * math.log(max_rank))))
        ranks.add(min(max(rank, 1), max_rank))

    linear_steps = max(min(num_auto_ranks, max_rank), 2)
    for step_idx in range(linear_steps):
        frac = step_idx / max(linear_steps - 1, 1)
        rank = int(round(1 + frac * (max_rank - 1)))
        ranks.add(min(max(rank, 1), max_rank))

    if math.isfinite(target_rank):
        target_int = int(round(target_rank))
        for delta in (
            -32,
            -24,
            -16,
            -12,
            -8,
            -4,
            -2,
            -1,
            0,
            1,
            2,
            4,
            8,
            12,
            16,
            24,
            32,
        ):
            rank = target_int + delta
            if 1 <= rank <= max_rank:
                ranks.add(rank)
        for scale in (0.5, 0.67, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5):
            rank = int(round(target_rank * scale))
            if 1 <= rank <= max_rank:
                ranks.add(rank)

    return sorted(ranks)


def resolve_rank_sweep(
    explicit_ranks: Sequence[int],
    *,
    max_rank: int,
    target_rank: float,
    num_auto_ranks: int,
) -> list[int]:
    if explicit_ranks:
        ranks = sorted({int(rank) for rank in explicit_ranks})
    else:
        ranks = build_auto_rank_sweep(
            max_rank=max_rank,
            target_rank=target_rank,
            num_auto_ranks=num_auto_ranks,
        )

    if not ranks:
        raise ValueError("Rank sweep is empty")
    invalid_ranks = [rank for rank in ranks if not 1 <= rank <= max_rank]
    if invalid_ranks:
        raise ValueError(
            f"Ranks must lie in [1, {max_rank}], got invalid values {invalid_ranks}"
        )
    return ranks


def build_plot_rank_sweep(
    *,
    max_rank: int,
    num_points: int | None,
) -> list[int]:
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if num_points is None or num_points >= max_rank:
        return list(range(1, max_rank + 1))
    if num_points <= 0:
        raise ValueError("num_plot_points must be positive")
    if num_points == 1:
        return [1]
    max_offset = max_rank - 1
    return [
        1 + (step_idx * max_offset) // (num_points - 1)
        for step_idx in range(num_points)
    ]


def parameter_axis_scale(*, units: str) -> float:
    if units == "ratio":
        return 1.0
    if units == "percentage":
        return 100.0
    raise ValueError(f"Unsupported x-axis units: {units}")


def parameter_axis_title(*, units: str) -> str:
    if units == "ratio":
        return "# of Parameters (ratio to TPR)"
    if units == "percentage":
        return "# of Parameters (% of TPR)"
    raise ValueError(f"Unsupported x-axis units: {units}")


def format_ratio_tick_label(value: float) -> str:
    label = f"{round(value, 2)}"
    if label == "1.24":
        return "1.25"
    if label == "1.49":
        return "1.5"
    if label == "1.74":
        return "1.75"
    if label == "1.99":
        return "2.0"
    return label


def format_parameter_tick_label(value: float, *, units: str) -> str:
    ratio_label = format_ratio_tick_label(value)
    if units == "ratio":
        return ratio_label
    if units == "percentage":
        percentage_value = float(ratio_label) * 100.0
        if percentage_value.is_integer():
            return f"{int(percentage_value)}%"
        return f"{percentage_value}%"
    raise ValueError(f"Unsupported x-axis units: {units}")


def infer_output_path(
    linear_probe_path: Path,
    tpr_probe_path: Path,
    *,
    use_effective_tpr_weights: bool,
) -> Path:
    source_tag = "_effective_tpr_weights" if use_effective_tpr_weights else ""
    return linear_probe_path.with_name(
        f"{linear_probe_path.stem}_svd_vs_{tpr_probe_path.stem}{source_tag}.json"
    )


def infer_plot_output_path(
    output_path: Path,
) -> Path:
    return output_path.with_name(f"{output_path.stem}_accuracy_curve.pdf")


def find_first_rank_at_accuracy_threshold(
    svd_results: Sequence[dict],
    *,
    accuracy_threshold: float,
) -> dict | None:
    eligible_results = [
        result
        for result in sorted(svd_results, key=lambda item: int(item["parameter_count"]))
        if float(result["accuracy"]) >= accuracy_threshold
    ]
    if not eligible_results:
        return None
    return eligible_results[0]


def plot_svd_accuracy_curve(
    *,
    svd_results: Sequence[dict],
    target_tpr_parameter_count: int,
    output_path: Path,
    target_ranks: Sequence[int] | None = None,
    tick_positions: Sequence[float] | None = None,
    x_axis_units: str = "ratio",
    accuracy_threshold: float = 0.99,
) -> dict[str, float | int | None]:
    if target_tpr_parameter_count <= 0:
        raise ValueError("target_tpr_parameter_count must be positive")
    if not svd_results:
        raise ValueError("svd_results must be non-empty")

    sorted_results = sorted(
        svd_results, key=lambda result: int(result["parameter_count"])
    )
    if target_ranks is None:
        tick_ranks = [int(result["rank"]) for result in sorted_results]
        tick_positions = [
            float(result["parameter_count"]) / float(target_tpr_parameter_count)
            for result in sorted_results
        ]
    else:
        tick_ranks = [int(rank) for rank in target_ranks]
        if tick_positions is None:
            raise ValueError(
                "tick_positions must be provided when target_ranks are specified"
            )
        tick_positions = [float(position) for position in tick_positions]
    if tick_positions is None or len(tick_positions) != len(tick_ranks):
        raise ValueError("tick_positions must have the same length as target_ranks")

    curve_parameter_percentages = [
        float(result["parameter_count"]) / float(target_tpr_parameter_count)
        for result in sorted_results
    ]
    curve_accuracies = [float(result["accuracy"]) for result in sorted_results]
    x_scale = parameter_axis_scale(units=x_axis_units)
    curve_x_values = [x_scale * value for value in curve_parameter_percentages]
    tick_x_positions = [x_scale * value for value in tick_positions]

    first_rank_at_threshold = find_first_rank_at_accuracy_threshold(
        sorted_results,
        accuracy_threshold=accuracy_threshold,
    )
    closest_rank_by_parameter_count = min(
        sorted_results,
        key=lambda result: abs(
            int(result["parameter_count"]) - int(target_tpr_parameter_count)
        ),
    )
    first_rank_threshold_pct = (
        None
        if first_rank_at_threshold is None
        else 1
        * float(first_rank_at_threshold["parameter_count"])
        / float(target_tpr_parameter_count)
    )
    first_rank_threshold_x = (
        None if first_rank_threshold_pct is None else x_scale * first_rank_threshold_pct
    )

    paper_width_in = 3.5
    paper_height_in = 1.8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(paper_width_in, paper_height_in), constrained_layout=True
    )
    ax.plot(
        curve_x_values,
        curve_accuracies,
        color="#2C6593",
        marker="o",
        linewidth=1.2,
        markersize=2.8,
        label="Rank-K SVD",
    )

    ax.axvline(
        x_scale,
        color="#C62828",
        linestyle="--",
        linewidth=0.9,
        label="TPR match",
    )
    if first_rank_threshold_x is not None:
        ax.axvline(
            first_rank_threshold_x,
            color="#1565C0",
            linestyle="--",
            linewidth=0.9,
            label="99% acc",
        )

    x_reference_positions = curve_x_values + tick_x_positions
    x_max = max(
        max(x_reference_positions),
        x_scale,
        first_rank_threshold_x if first_rank_threshold_x is not None else 0.0,
    )
    x_margin = max(0.025, 0.04 * x_max)
    ax.set_xlim(
        left=max(0.0, min(x_reference_positions) - x_margin),
        right=x_max + x_margin,
    )

    y_min = min(curve_accuracies)
    y_max = max(curve_accuracies)
    y_padding = max(0.005, 0.08 * max(y_max - y_min, 0.01))
    ax.set_ylim(max(0.0, y_min - y_padding), min(1.0, y_max + y_padding))

    ax.set_xlabel(parameter_axis_title(units=x_axis_units))
    ax.set_ylabel("Accuracy")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.85)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    tick_labels = []
    for percentage, rank in zip(tick_positions, tick_ranks, strict=True):
        percentage_label = format_parameter_tick_label(percentage, units=x_axis_units)
        tick_labels.append(f"{percentage_label}\nk={rank}")
    ax.set_xticks(tick_x_positions, tick_labels)
    ax.tick_params(axis="x", labelsize=5.6, pad=1.5, length=2.5, width=0.6)
    ax.tick_params(axis="y", labelsize=7, pad=1.5, length=2.5, width=0.6)
    for label in ax.get_xticklabels():
        label.set_multialignment("center")
        label.set_linespacing(0.9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        borderpad=0.15,
        labelspacing=0.2,
    )
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    plt.close(fig)

    return {
        "accuracy_threshold": accuracy_threshold,
        "first_rank_at_threshold": (
            None
            if first_rank_at_threshold is None
            else int(first_rank_at_threshold["rank"])
        ),
        "first_rank_at_threshold_parameter_count": (
            None
            if first_rank_at_threshold is None
            else int(first_rank_at_threshold["parameter_count"])
        ),
        "first_rank_at_threshold_parameter_pct_of_tpr": (
            None
            if first_rank_threshold_pct is None
            else float(first_rank_threshold_pct)
        ),
        "closest_rank_by_parameter_count": int(closest_rank_by_parameter_count["rank"]),
        "x_axis_units": x_axis_units,
        "num_curve_points": int(len(curve_parameter_percentages)),
        "num_x_tick_labels": int(len(tick_x_positions)),
    }


def render_result_rows(
    *,
    full_metrics: dict,
    tpr_metrics: dict | None,
    svd_results: list[dict],
    closest_rank: int,
    full_method_label: str,
) -> list[str]:
    header = (
        f"{'method':<16} {'rank':>6} {'params':>10} "
        f"{'param/tpr':>10} {'energy':>8} {'acc':>10} {'macro_f1':>10}"
    )
    rows = [header, "-" * len(header)]
    rows.append(
        f"{full_method_label:<16} {'full':>6} "
        f"{int(full_metrics['parameter_count']):>10d} "
        f"{full_metrics['parameter_ratio_to_tpr']:>10.4f} "
        f"{1.0:>8.4f} "
        f"{full_metrics['accuracy']:>10.4f} "
        f"{full_metrics['macro_f1']:>10.4f}"
    )
    for result in svd_results:
        method_label = "svd"
        if int(result["rank"]) == int(closest_rank):
            method_label = "svd*"
        rows.append(
            f"{method_label:<16} {int(result['rank']):>6d} "
            f"{int(result['parameter_count']):>10d} "
            f"{result['parameter_ratio_to_tpr']:>10.4f} "
            f"{result['frobenius_energy_fraction']:>8.4f} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_f1']:>10.4f}"
        )
    if tpr_metrics is not None:
        rows.append(
            f"{'tpr':<16} {'-':>6} "
            f"{int(tpr_metrics['parameter_count']):>10d} "
            f"{tpr_metrics['parameter_ratio_to_tpr']:>10.4f} "
            f"{'-':>8} "
            f"{tpr_metrics['accuracy']:>10.4f} "
            f"{tpr_metrics['macro_f1']:>10.4f}"
        )
    return rows


def evaluate_rank_sweep(
    *,
    model,
    full_module_name: str,
    full_weight_flat: Tensor,
    full_bias_flat: Tensor | None,
    tpr_probe: torch.nn.Module | None,
    tpr_module_name: str | None,
    ranks: Sequence[int],
    singular_values: Tensor,
    encoder: Tensor,
    decoder: Tensor,
    svd_bias_parameter_count: int,
    games_tokens: Tensor,
    games_raw: Sequence[Sequence[int]],
    batch_size: int,
    pos_start: int,
    pos_end: int,
    device: torch.device,
    square_mask: Tensor,
) -> tuple[dict, dict | None, list[dict]]:
    modules = [full_module_name]
    if (
        tpr_probe is not None
        and tpr_module_name is not None
        and tpr_module_name not in modules
    ):
        modules.append(tpr_module_name)

    full_weight_flat = full_weight_flat.to(device)
    if full_bias_flat is not None:
        full_bias_flat = full_bias_flat.to(device)
    max_rank = max(ranks)
    encoder = encoder[:, :max_rank].to(device)
    decoder = decoder[:max_rank].to(device)
    singular_values = singular_values.to(device)
    square_mask = square_mask.to(device=device, dtype=torch.bool)

    full_accumulator = MetricAccumulator()
    tpr_accumulator = MetricAccumulator() if tpr_probe is not None else None
    svd_accumulators = {int(rank): MetricAccumulator() for rank in ranks}

    if tpr_probe is not None:
        tpr_probe.eval()

    with torch.inference_mode():
        for batch_start in tqdm(
            range(0, len(games_raw), batch_size),
            desc="evaluating",
        ):
            batch_end = min(batch_start + batch_size, len(games_raw))
            batch_tokens = games_tokens[batch_start:batch_end].to(device)
            batch_raw = games_raw[batch_start:batch_end]

            state_stack = build_state_stack(batch_raw)[:, pos_start:pos_end].to(device)
            targets_one_hot = state_stack_to_one_hot_threeway(state_stack)[0]

            with record_activations(model, modules) as cache:
                model(batch_tokens[:, :-1])

            full_residual = cache[full_module_name][0][:, pos_start:pos_end].to(device)
            full_logits_flat = torch.einsum(
                "bpd,do->bpo", full_residual, full_weight_flat
            )
            if full_bias_flat is not None:
                full_logits_flat = full_logits_flat + full_bias_flat.view(1, 1, -1)
            full_logits = full_logits_flat.reshape(
                full_logits_flat.shape[0],
                full_logits_flat.shape[1],
                BOARD_ROWS,
                BOARD_COLS,
                BOARD_LABEL_OPTIONS,
            )
            full_accumulator.update(
                logits=full_logits,
                targets_one_hot=targets_one_hot,
                square_mask=square_mask,
            )

            compressed_residual = torch.einsum("bpd,dk->bpk", full_residual, encoder)
            for rank in ranks:
                logits_flat = torch.einsum(
                    "bpk,ko->bpo",
                    compressed_residual[:, :, :rank],
                    decoder[:rank],
                )
                if full_bias_flat is not None:
                    logits_flat = logits_flat + full_bias_flat.view(1, 1, -1)
                svd_logits = logits_flat.reshape(
                    logits_flat.shape[0],
                    logits_flat.shape[1],
                    BOARD_ROWS,
                    BOARD_COLS,
                    BOARD_LABEL_OPTIONS,
                )
                svd_accumulators[int(rank)].update(
                    logits=svd_logits,
                    targets_one_hot=targets_one_hot,
                    square_mask=square_mask,
                )

            if (
                tpr_probe is not None
                and tpr_accumulator is not None
                and tpr_module_name is not None
            ):
                if tpr_module_name == full_module_name:
                    tpr_residual = full_residual
                else:
                    tpr_residual = cache[tpr_module_name][0][:, pos_start:pos_end].to(
                        device
                    )
                tpr_logits = tpr_probe(tpr_residual)
                tpr_accumulator.update(
                    logits=tpr_logits,
                    targets_one_hot=targets_one_hot,
                    square_mask=square_mask,
                )

    full_metrics = full_accumulator.finalize()
    tpr_metrics = tpr_accumulator.finalize() if tpr_accumulator is not None else None

    total_energy = singular_values.square().sum()
    svd_results = []
    for rank in ranks:
        rank = int(rank)
        svd_metrics = svd_accumulators[rank].finalize()
        svd_metrics["rank"] = rank
        svd_metrics["parameter_count"] = compute_rank_parameter_count(
            rank=rank,
            d_model=int(full_weight_flat.shape[0]),
            num_outputs=int(full_weight_flat.shape[1]),
            bias_parameter_count=svd_bias_parameter_count,
        )
        svd_metrics["frobenius_energy_fraction"] = float(
            singular_values[:rank].square().sum().item() / total_energy.item()
        )
        svd_results.append(svd_metrics)
    return full_metrics, tpr_metrics, svd_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-probe-path", required=True, help="Path to a saved linear probe."
    )
    parser.add_argument(
        "--tpr-probe-path",
        required=True,
        help=("Path to a saved activation-backed TPR probe."),
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional model checkpoint override. Defaults to the TPR probe config, then the linear probe config.",
    )
    parser.add_argument(
        "--data-path",
        default="test_data",
        help="Optional dataset path override. Defaults to the TPR probe config, then the linear probe config.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional JSON output path. Defaults next to the linear probe checkpoint.",
    )
    parser.add_argument(
        "--plot-output-path",
        help="Optional output path for the SVD accuracy curve plot.",
    )
    parser.add_argument("--device", default=LinearProbeSVDBaselineConfig.device)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default=LinearProbeSVDBaselineConfig.split,
        help="Which split to evaluate on.",
    )
    parser.add_argument("--valid-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--pos-start", type=int)
    parser.add_argument(
        "--pos-end",
        type=int,
        help="Exclusive end position. Non-positive values are treated as the model block size.",
    )
    parser.add_argument(
        "--rank",
        dest="ranks",
        type=int,
        action="append",
        default=None,
        help="Explicit SVD rank to evaluate. May be passed multiple times.",
    )
    parser.add_argument(
        "--num-auto-ranks",
        type=int,
        default=LinearProbeSVDBaselineConfig.num_auto_ranks,
        help="Number of coarse anchor points used when ranks are auto-generated.",
    )
    parser.add_argument(
        "--num-plot-points",
        type=int,
        default=LinearProbeSVDBaselineConfig.num_plot_points,
        help=(
            "Number of SVD points to evaluate and draw on the curve. "
            "Defaults to every rank; x-axis ticks stay fixed."
        ),
    )
    parser.add_argument(
        "--x-axis-units",
        choices=("ratio", "percentage"),
        default=LinearProbeSVDBaselineConfig.x_axis_units,
        help="Render the x-axis as a ratio to TPR or as a percentage of TPR.",
    )
    parser.add_argument(
        "--skip-tpr-eval",
        action="store_true",
        help="Only use the TPR checkpoint for parameter-count matching; skip re-evaluating it.",
    )
    parser.add_argument(
        "--use-effective-tpr-weights",
        action="store_true",
        help=(
            "Use the TPR probe's effective linear weights as the SVD target instead "
            "of the saved linear probe weights."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.ranks = [1, 20, 40, 60, 80, 100, 120, 140, 160, 192]
    config = LinearProbeSVDBaselineConfig(
        linear_probe_path=args.linear_probe_path,
        tpr_probe_path=args.tpr_probe_path,
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        plot_output_path=args.plot_output_path,
        device=args.device,
        n_head=args.n_head,
        batch_size=args.batch_size,
        split=args.split,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        max_games=args.max_games,
        pos_start=args.pos_start,
        pos_end=args.pos_end,
        num_auto_ranks=args.num_auto_ranks,
        num_plot_points=args.num_plot_points,
        x_axis_units=args.x_axis_units,
        ranks=tuple(args.ranks or ()),
        skip_tpr_eval=args.skip_tpr_eval,
        use_effective_tpr_weights=args.use_effective_tpr_weights,
    )

    linear_probe_path = resolve_path(config.linear_probe_path)
    tpr_probe_path = resolve_path(config.tpr_probe_path)
    output_path = (
        resolve_path(config.output_path)
        if config.output_path is not None
        else infer_output_path(
            linear_probe_path=linear_probe_path,
            tpr_probe_path=tpr_probe_path,
            use_effective_tpr_weights=config.use_effective_tpr_weights,
        )
    )
    plot_output_path = (
        resolve_path(config.plot_output_path)
        if config.plot_output_path is not None
        else infer_plot_output_path(output_path)
    )

    device = torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    linear_probe, linear_layer, linear_module_name, linear_artifact = (
        load_linear_probe_artifact(
            probe_path=linear_probe_path,
            device=device,
        )
    )
    linear_weight_flat = flatten_linear_probe(linear_probe)
    linear_weight_matrix = linear_weight_flat.T.detach().cpu()

    linear_config = linear_artifact.get("config", {})
    if not isinstance(linear_config, dict):
        linear_config = {}

    loaded_tpr_probe = load_tpr_like_probe(
        probe_path=tpr_probe_path,
        linear_d_model=int(linear_weight_flat.shape[0]),
        device=device,
    )
    tpr_probe = loaded_tpr_probe.probe
    tpr_layer = loaded_tpr_probe.layer
    tpr_artifact = loaded_tpr_probe.artifact
    tpr_probe_kind = loaded_tpr_probe.probe_kind
    tpr_module_name = loaded_tpr_probe.module_name
    tpr_square_color_vectors = loaded_tpr_probe.square_color_vectors
    tpr_config = tpr_artifact.get("config", {})
    if not isinstance(tpr_config, dict):
        tpr_config = {}

    if int(linear_weight_flat.shape[0]) != int(tpr_probe.d_model):
        raise ValueError(
            "Linear probe and TPR probe have incompatible d_model sizes: "
            f"{int(linear_weight_flat.shape[0])} vs {int(tpr_probe.d_model)}"
        )

    svd_source_label = "linear_probe_weights"
    full_method_label = "linear_full"
    full_weight_flat = linear_weight_flat.detach()
    full_bias_flat = None
    full_module_name = linear_module_name
    full_weight_matrix = linear_weight_matrix
    svd_bias_parameter_count = 0
    if config.use_effective_tpr_weights:
        effective_weight_tensor, effective_bias = compute_tpr_effective_weights(
            tpr_probe
        )
        full_weight_flat = flatten_effective_weight_tensor(
            effective_weight_tensor
        ).detach()
        full_weight_matrix = full_weight_flat.T.detach().cpu()
        full_bias_flat = (
            flatten_output_bias(effective_bias).detach()
            if effective_bias is not None
            else None
        )
        full_module_name = tpr_module_name
        full_method_label = "tpr_eff_full"
        svd_source_label = "tpr_effective_linear_weights"
        svd_bias_parameter_count = (
            0 if full_bias_flat is None else int(full_bias_flat.numel())
        )
    svd_evaluation_space = "model_activation_stream"

    seed_value = int(
        choose_config_value(
            config.seed,
            tpr_config,
            linear_config,
            "seed",
            default=TPRProbeConfig.seed,
        )
    )
    seed_all(seed_value)

    checkpoint_path = resolve_path(
        choose_config_value(
            config.checkpoint,
            tpr_config,
            linear_config,
            "checkpoint",
            default=TPRProbeConfig.checkpoint,
        )
    )
    data_path = resolve_path(
        choose_config_value(
            config.data_path,
            tpr_config,
            linear_config,
            "data_path",
            default="test_data",
        )
    )
    n_head = int(
        choose_config_value(
            config.n_head,
            tpr_config,
            linear_config,
            "n_head",
            default=TPRProbeConfig.n_head,
        )
    )
    batch_size = int(
        choose_config_value(
            config.batch_size,
            tpr_config,
            linear_config,
            "batch_size",
            default=TPRProbeConfig.batch_size,
        )
    )
    valid_size = int(
        choose_config_value(
            config.valid_size,
            tpr_config,
            linear_config,
            "valid_size",
            default=ProbeConfig.valid_size,
        )
    )
    test_size = int(
        choose_config_value(
            config.test_size,
            tpr_config,
            linear_config,
            "test_size",
            default=ProbeConfig.test_size,
        )
    )
    max_games = choose_config_value(
        config.max_games,
        tpr_config,
        linear_config,
        "max_games",
        default=None,
    )
    raw_pos_start = int(
        choose_config_value(
            config.pos_start,
            tpr_config,
            linear_config,
            "pos_start",
            default=0,
        )
    )
    raw_pos_end = normalize_pos_end(
        choose_config_value(
            normalize_pos_end(config.pos_end),
            tpr_config,
            linear_config,
            "pos_end",
            default=None,
        )
    )

    model = load_model(
        {
            "model_path": str(checkpoint_path),
            "device": device,
            "load_weights": True,
            "n_head": n_head,
        }
    )
    convert_to_hooked_model(model)
    pos_start, pos_end = resolve_position_slice(
        raw_pos_start,
        raw_pos_end,
        model.get_block_size(),
    )

    split = load_probe_dataset(
        data_path=data_path,
        block_size=model.get_block_size(),
        valid_size=valid_size,
        test_size=test_size,
        seed=seed_value,
        max_games=max_games,
    )
    split_tokens_key = f"{config.split}_tokens"
    split_raw_key = f"{config.split}_raw"
    games_tokens = split[split_tokens_key]
    games_raw = split[split_raw_key]

    square_mask = torch.ones(BOARD_ROWS, BOARD_COLS, device=device, dtype=torch.bool)
    primary_square_count = int(square_mask.sum().item())
    evaluation_context = EvaluationContext(
        checkpoint=checkpoint_path,
        data_path=data_path,
        batch_size=batch_size,
        valid_size=valid_size,
        test_size=test_size,
        seed=seed_value,
        max_games=max_games,
        pos_start=pos_start,
        pos_end=pos_end,
        primary_square_count=primary_square_count,
        split=config.split,
        split_tokens_key=split_tokens_key,
        split_raw_key=split_raw_key,
    )

    encoder, decoder, singular_values = compute_truncated_svd_factors(
        full_weight_matrix
    )
    target_tpr_parameter_count = int(
        tpr_artifact.get(
            "probe_parameter_count",
            sum(parameter.numel() for parameter in tpr_probe.parameters()),
        )
    )
    target_rank_from_tpr = (target_tpr_parameter_count - svd_bias_parameter_count) / (
        int(full_weight_matrix.shape[1]) + int(full_weight_matrix.shape[0])
    )
    max_rank = int(singular_values.numel())
    target_ranks = resolve_rank_sweep(
        config.ranks,
        max_rank=max_rank,
        target_rank=target_rank_from_tpr,
        num_auto_ranks=config.num_auto_ranks,
    )
    evaluated_ranks = build_plot_rank_sweep(
        max_rank=max_rank,
        num_points=config.num_plot_points,
    )
    tick_positions = [
        compute_rank_parameter_count(
            rank=rank,
            d_model=int(full_weight_matrix.shape[1]),
            num_outputs=int(full_weight_matrix.shape[0]),
            bias_parameter_count=svd_bias_parameter_count,
        )
        / target_tpr_parameter_count
        for rank in target_ranks
    ]

    print("Linear probe SVD config:")
    print(json.dumps(asdict(config), indent=2))
    print("Resolved evaluation context:")
    print(
        json.dumps(
            {
                "linear_probe_path": str(linear_probe_path),
                "linear_layer": linear_layer,
                "linear_module_name": linear_module_name,
                "tpr_probe_path": str(tpr_probe_path),
                "tpr_probe_kind": tpr_probe_kind,
                "tpr_layer": tpr_layer,
                "tpr_module_name": tpr_module_name,
                "svd_weight_source": svd_source_label,
                "svd_full_method_label": full_method_label,
                "svd_evaluation_space": svd_evaluation_space,
                "svd_module_name": full_module_name,
                "checkpoint": str(evaluation_context.checkpoint),
                "data_path": str(evaluation_context.data_path),
                "split": evaluation_context.split,
                "batch_size": evaluation_context.batch_size,
                "seed": evaluation_context.seed,
                "max_games": evaluation_context.max_games,
                "pos_start": evaluation_context.pos_start,
                "pos_end": evaluation_context.pos_end,
                "primary_square_count": evaluation_context.primary_square_count,
                "split_size": len(games_raw),
                "d_model": int(full_weight_matrix.shape[1]),
                "num_outputs": int(full_weight_matrix.shape[0]),
                "linear_parameter_count": int(linear_weight_matrix.numel()),
                "svd_source_parameter_count": int(full_weight_matrix.numel())
                + svd_bias_parameter_count,
                "svd_bias_parameter_count": svd_bias_parameter_count,
                "tpr_parameter_count": target_tpr_parameter_count,
                "target_rank_from_tpr_params": target_rank_from_tpr,
                "ranks": target_ranks,
                "num_plot_points_requested": config.num_plot_points,
                "x_axis_units": config.x_axis_units,
                "num_evaluated_ranks": len(evaluated_ranks),
                "output_path": str(output_path),
                "plot_output_path": str(plot_output_path),
            },
            indent=2,
        )
    )
    if not config.use_effective_tpr_weights and (
        linear_layer != tpr_layer or linear_module_name != tpr_module_name
    ):
        print(
            "Note: the linear probe and TPR probe do not use the same cached activation stream. "
            "They will still be evaluated on the same split, but from their own saved modules."
        )
    full_metrics, tpr_metrics, svd_results = evaluate_rank_sweep(
        model=model,
        full_module_name=full_module_name,
        full_weight_flat=full_weight_flat.detach(),
        full_bias_flat=None if full_bias_flat is None else full_bias_flat.detach(),
        tpr_probe=None if config.skip_tpr_eval else tpr_probe,
        tpr_module_name=(None if config.skip_tpr_eval else tpr_module_name),
        ranks=evaluated_ranks,
        singular_values=singular_values,
        encoder=encoder,
        decoder=decoder,
        svd_bias_parameter_count=svd_bias_parameter_count,
        games_tokens=games_tokens,
        games_raw=games_raw,
        batch_size=evaluation_context.batch_size,
        pos_start=evaluation_context.pos_start,
        pos_end=evaluation_context.pos_end,
        device=device,
        square_mask=square_mask,
    )
    full_parameter_count = int(full_weight_matrix.numel()) + svd_bias_parameter_count
    full_metrics["parameter_count"] = full_parameter_count
    full_metrics["parameter_ratio_to_tpr"] = (
        full_parameter_count / target_tpr_parameter_count
    )
    if tpr_metrics is not None:
        tpr_metrics["parameter_count"] = target_tpr_parameter_count
        tpr_metrics["parameter_ratio_to_tpr"] = 1.0

    for result in svd_results:
        result["parameter_ratio_to_tpr"] = (
            result["parameter_count"] / target_tpr_parameter_count
        )
        result["parameter_delta_to_tpr"] = (
            int(result["parameter_count"]) - target_tpr_parameter_count
        )
    closest_svd_result = min(
        svd_results,
        key=lambda result: abs(
            int(result["parameter_count"]) - target_tpr_parameter_count
        ),
    )
    closest_rank = int(closest_svd_result["rank"])

    plot_summary = plot_svd_accuracy_curve(
        svd_results=svd_results,
        target_tpr_parameter_count=target_tpr_parameter_count,
        output_path=plot_output_path,
        target_ranks=target_ranks,
        tick_positions=tick_positions,
        x_axis_units=config.x_axis_units,
    )

    output_payload = {
        "linear_probe_path": str(linear_probe_path),
        "tpr_probe_path": str(tpr_probe_path),
        "plot_path": str(plot_output_path),
        "svd_weight_source": svd_source_label,
        "svd_full_method_label": full_method_label,
        "svd_evaluation_space": svd_evaluation_space,
        "evaluation_context": {
            "checkpoint": str(evaluation_context.checkpoint),
            "data_path": str(evaluation_context.data_path),
            "split": evaluation_context.split,
            "split_size": len(games_raw),
            "batch_size": evaluation_context.batch_size,
            "seed": evaluation_context.seed,
            "max_games": evaluation_context.max_games,
            "pos_start": evaluation_context.pos_start,
            "pos_end": evaluation_context.pos_end,
            "primary_square_count": evaluation_context.primary_square_count,
            "linear_layer": linear_layer,
            "linear_module_name": linear_module_name,
            "tpr_probe_kind": tpr_probe_kind,
            "tpr_layer": tpr_layer,
            "tpr_module_name": tpr_module_name,
            "svd_module_name": full_module_name,
            "svd_evaluation_space": svd_evaluation_space,
        },
        "parameter_matching": {
            "d_model": int(full_weight_matrix.shape[1]),
            "num_outputs": int(full_weight_matrix.shape[0]),
            "linear_parameter_count": int(linear_weight_matrix.numel()),
            "svd_source_parameter_count": full_parameter_count,
            "svd_bias_parameter_count": svd_bias_parameter_count,
            "tpr_parameter_count": target_tpr_parameter_count,
            "target_rank_from_tpr_params": target_rank_from_tpr,
            "closest_rank_by_parameter_count": int(closest_rank),
            "closest_rank_parameter_count": int(closest_svd_result["parameter_count"]),
            "closest_rank_parameter_delta_to_tpr": int(
                closest_svd_result["parameter_count"] - target_tpr_parameter_count
            ),
        },
        "tpr": tpr_metrics,
        "svd_sweep": svd_results,
        "closest_rank_result": closest_svd_result,
        "svd_plot_summary": plot_summary,
        "linear_artifact_metrics": linear_artifact.get("metrics"),
        "tpr_artifact_metrics": tpr_artifact.get("metrics"),
    }
    if config.use_effective_tpr_weights:
        output_payload["effective_tpr_full"] = full_metrics
    else:
        output_payload["linear_full"] = full_metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    if len(svd_results) <= 25:
        displayed_svd_results = svd_results
    else:
        displayed_rank_set = {int(rank) for rank in target_ranks}
        displayed_rank_set.add(int(closest_rank))
        displayed_svd_results = [
            result
            for result in svd_results
            if int(result["rank"]) in displayed_rank_set
        ]

    print()
    print(
        "\n".join(
            render_result_rows(
                full_metrics=full_metrics,
                tpr_metrics=tpr_metrics,
                svd_results=displayed_svd_results,
                closest_rank=closest_rank,
                full_method_label=full_method_label,
            )
        )
    )
    print()
    print(
        "Closest parameter-matched SVD rank "
        f"{closest_rank}: accuracy={closest_svd_result['accuracy']:.4f} "
        f"macro_f1={closest_svd_result['macro_f1']:.4f} "
        f"params={closest_svd_result['parameter_count']}"
    )
    if tpr_metrics is not None:
        print(
            "TPR eval score: "
            f"loss={tpr_metrics['loss']:.4f} "
            f"accuracy={tpr_metrics['accuracy']:.4f} "
            f"macro_f1={tpr_metrics['macro_f1']:.4f} "
            f"params={tpr_metrics['parameter_count']}"
        )
    print("Closest-rank square accuracy:")
    print(
        format_square_accuracy_board(
            torch.tensor(closest_svd_result["square_accuracy"])
        )
    )
    if tpr_metrics is not None:
        print("TPR square accuracy:")
        print(
            format_square_accuracy_board(torch.tensor(tpr_metrics["square_accuracy"]))
        )
    print(f"Wrote results to {output_path}")
    print(f"Wrote SVD accuracy curve to {plot_output_path}")


if __name__ == "__main__":
    main()
