"""Compare TPR and multilinear TPR probe weights against the same linear probe."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.nn.functional as F
import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parent.parent

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from mpl_toolkits.axes_grid1 import ImageGrid  # noqa: E402

BOARD_ROWS = 8
BOARD_COLS = 8
BOARD_LABEL_OPTIONS = 3
ROW_LABELS = "ABCDEFGH"
STATE_LABELS = ("empty", "opponent", "current")
STATE_DISPLAY_LABELS = ("Empty", "Opponent", "Current")
DIRECTION_DIFFERENCE_INDEX_PAIRS = ((0, 1), (1, 2), (2, 0))
DIRECTION_DIFFERENCE_LABELS = tuple(
    f"{STATE_LABELS[left_idx]}-{STATE_LABELS[right_idx]}"
    for left_idx, right_idx in DIRECTION_DIFFERENCE_INDEX_PAIRS
)
DIRECTION_DIFFERENCE_DISPLAY_LABELS = tuple(
    f"{STATE_DISPLAY_LABELS[left_idx]} - {STATE_DISPLAY_LABELS[right_idx]}"
    for left_idx, right_idx in DIRECTION_DIFFERENCE_INDEX_PAIRS
)

PAPER_RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def configure_paper_style() -> None:
    plt.rcParams.update(PAPER_RC_PARAMS)


def load_linear_probe_weights(
    probe_path: Path, device: torch.device
) -> tuple[Tensor, dict]:
    artifact = torch.load(probe_path, map_location=device)
    probe = artifact["probe"].to(device=device, dtype=torch.float32)
    if probe.shape[0] != 1:
        raise ValueError(
            f"Expected leading probe dimension of size 1 in {probe_path}, got {probe.shape}"
        )
    return probe[0], artifact


def load_tpr_effective_weights(
    probe_path: Path, device: torch.device
) -> tuple[Tensor, dict, Tensor | None]:
    artifact = torch.load(probe_path, map_location=device)
    state_dict = artifact["probe_state_dict"]
    binding_map = state_dict["binding_map"].to(device=device, dtype=torch.float32)
    role_embeddings = state_dict["role_embeddings"].to(
        device=device, dtype=torch.float32
    )
    filler_embeddings = state_dict["filler_embeddings"].to(
        device=device, dtype=torch.float32
    )
    effective_weights = torch.einsum(
        "drf,xyr,cf->dxyc",
        binding_map,
        role_embeddings,
        filler_embeddings,
    )
    bias = state_dict.get("bias")
    if bias is not None:
        bias = bias.to(device=device, dtype=torch.float32)
    return effective_weights, artifact, bias


def mean_center_weights(weights: Tensor) -> Tensor:
    if weights.shape[-1] != len(STATE_LABELS):
        raise ValueError(
            "Mean-centering expects the final dimension to match the three probe "
            f"states, got shape {tuple(weights.shape)}"
        )
    return weights - weights.mean(dim=-1, keepdim=True)


def summarize_tensor(values: Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def load_multilinear_effective_weights(
    probe_path: Path, device: torch.device
) -> tuple[Tensor, dict, Tensor | None]:
    artifact = torch.load(probe_path, map_location=device)
    state_dict = artifact["probe_state_dict"]
    binding_map = state_dict["binding_map"].to(device=device, dtype=torch.float32)
    row_embeddings = state_dict["row_embeddings"].to(device=device, dtype=torch.float32)
    col_embeddings = state_dict["col_embeddings"].to(device=device, dtype=torch.float32)
    color_embeddings = state_dict["color_embeddings"].to(
        device=device, dtype=torch.float32
    )
    effective_weights = torch.einsum(
        "drck,xr,yc,ok->dxyo",
        binding_map,
        row_embeddings,
        col_embeddings,
        color_embeddings,
    )
    bias = state_dict.get("bias")
    if bias is not None:
        bias = bias.to(device=device, dtype=torch.float32)
    return effective_weights, artifact, bias


def resolve_comparison_metadata():
    return (
        STATE_LABELS,
        STATE_DISPLAY_LABELS,
        "states",
        "state",
        "absolute_cosine_similarity",
    )


def apply_display_mask(board: Tensor, is_no_center: bool) -> Tensor:
    board = board.detach().cpu().clone()
    if is_no_center:
        mask = build_display_mask(is_no_center=True)
        board[~mask] = float("nan")
    return board


def compute_abs_cosine_similarity(
    linear_weights: Tensor, tpr_weights: Tensor
) -> Tensor:
    return F.cosine_similarity(
        linear_weights.permute(1, 2, 3, 0),
        tpr_weights.permute(1, 2, 3, 0),
        dim=-1,
        eps=1e-12,
    ).abs()


def compare_multilinear_probe_pair(
    linear_probe_path: Path,
    multilinear_probe_path: Path,
    device: torch.device,
    compare_direction_differences: bool = False,
    mean_center_before_comparison: bool = False,
) -> tuple[dict, list[dict[str, float | int | str | None]]]:
    linear_weights, linear_artifact = load_linear_probe_weights(
        linear_probe_path, device
    )
    multilinear_weights, multilinear_artifact, multilinear_bias = (
        load_multilinear_effective_weights(multilinear_probe_path, device)
    )

    if linear_weights.shape != multilinear_weights.shape:
        raise ValueError(
            f"Shape mismatch between {linear_probe_path} {tuple(linear_weights.shape)} "
            f"and {multilinear_probe_path} {tuple(multilinear_weights.shape)}"
        )

    linear_layer = int(linear_artifact["layer"])
    multilinear_layer = int(multilinear_artifact["layer"])
    if linear_layer != multilinear_layer:
        raise ValueError(
            f"Layer mismatch: linear layer {linear_layer}, multilinear layer {multilinear_layer}"
        )

    (
        comparison_labels,
        comparison_display_labels,
        comparison_mode,
        comparison_label_kind,
        similarity_metric,
    ) = resolve_comparison_metadata()
    if mean_center_before_comparison:
        linear_weights = mean_center_weights(linear_weights)
        multilinear_weights = mean_center_weights(multilinear_weights)
    if compare_direction_differences:
        linear_weights = compute_direction_differences(linear_weights)
        multilinear_weights = compute_direction_differences(multilinear_weights)
        if multilinear_bias is not None:
            multilinear_bias = compute_direction_differences(multilinear_bias)

    abs_cosine_similarity = compute_abs_cosine_similarity(
        linear_weights, multilinear_weights
    )
    linear_norms = linear_weights.norm(dim=0)
    multilinear_norms = multilinear_weights.norm(dim=0)

    by_state = {}
    abs_cosine_boards = {}
    linear_norm_boards = {}
    multilinear_norm_boards = {}
    bias_boards = {}
    for state_idx, state_label in enumerate(comparison_labels):
        state_abs_cosine = abs_cosine_similarity[:, :, state_idx]
        state_linear_norms = linear_norms[:, :, state_idx]
        state_multilinear_norms = multilinear_norms[:, :, state_idx]
        by_state[state_label] = {
            "abs_cosine_similarity": summarize_tensor(state_abs_cosine),
            "linear_norm": summarize_tensor(state_linear_norms),
            "multilinear_effective_norm": summarize_tensor(state_multilinear_norms),
        }
        abs_cosine_boards[state_label] = state_abs_cosine.tolist()
        linear_norm_boards[state_label] = state_linear_norms.tolist()
        multilinear_norm_boards[state_label] = state_multilinear_norms.tolist()
        if multilinear_bias is not None:
            state_bias = multilinear_bias[:, :, state_idx]
            by_state[state_label]["multilinear_bias"] = summarize_tensor(state_bias)
            bias_boards[state_label] = state_bias.tolist()

    result = {
        "layer": linear_layer,
        "linear_probe_path": str(linear_probe_path),
        "multilinear_probe_path": str(multilinear_probe_path),
        "multilinear_row_dim": int(multilinear_artifact["row_dim"]),
        "multilinear_col_dim": int(multilinear_artifact["col_dim"]),
        "multilinear_color_dim": int(multilinear_artifact["color_dim"]),
        "multilinear_use_bias": bool(multilinear_artifact.get("use_bias", False)),
        "comparison_mode": comparison_mode,
        "comparison_label_kind": comparison_label_kind,
        "comparison_labels": list(comparison_labels),
        "comparison_display_labels": list(comparison_display_labels),
        "mean_center_before_comparison": mean_center_before_comparison,
        "similarity_metric": similarity_metric,
        "overall": {
            "abs_cosine_similarity": summarize_tensor(abs_cosine_similarity),
            "linear_norm": summarize_tensor(linear_norms),
            "multilinear_effective_norm": summarize_tensor(multilinear_norms),
        },
        "by_state": by_state,
        "abs_cosine_similarity_boards": abs_cosine_boards,
        "linear_norm_boards": linear_norm_boards,
        "multilinear_effective_norm_boards": multilinear_norm_boards,
        "multilinear_bias_boards": (
            bias_boards if multilinear_bias is not None else None
        ),
    }
    return result


def compare_tpr_probe_pair(
    linear_probe_path: Path,
    tpr_probe_path: Path,
    device: torch.device,
    compare_direction_differences: bool = False,
    mean_center_before_comparison: bool = False,
) -> tuple[dict, list[dict[str, float | int | str | None]]]:
    linear_weights, linear_artifact = load_linear_probe_weights(
        linear_probe_path, device
    )
    tpr_weights, tpr_artifact, tpr_bias = load_tpr_effective_weights(
        tpr_probe_path, device
    )

    if linear_weights.shape != tpr_weights.shape:
        raise ValueError(
            f"Shape mismatch between {linear_probe_path} {tuple(linear_weights.shape)} "
            f"and {tpr_probe_path} {tuple(tpr_weights.shape)}"
        )

    linear_layer = int(linear_artifact["layer"])
    tpr_layer = int(tpr_artifact["layer"])
    if linear_layer != tpr_layer:
        raise ValueError(
            f"Layer mismatch: linear layer {linear_layer}, TPR layer {tpr_layer}"
        )

    (
        comparison_labels,
        comparison_display_labels,
        comparison_mode,
        comparison_label_kind,
        similarity_metric,
    ) = resolve_comparison_metadata()
    if mean_center_before_comparison:
        linear_weights = mean_center_weights(linear_weights)
        tpr_weights = mean_center_weights(tpr_weights)
    if compare_direction_differences:
        linear_weights = compute_direction_differences(linear_weights)
        tpr_weights = compute_direction_differences(tpr_weights)
        if tpr_bias is not None:
            tpr_bias = compute_direction_differences(tpr_bias)

    abs_cosine_similarity = compute_abs_cosine_similarity(linear_weights, tpr_weights)
    linear_norms = linear_weights.norm(dim=0)
    tpr_norms = tpr_weights.norm(dim=0)

    by_state = {}
    abs_cosine_boards = {}
    linear_norm_boards = {}
    tpr_norm_boards = {}
    bias_boards = {}
    for state_idx, state_label in enumerate(comparison_labels):
        state_abs_cosine = abs_cosine_similarity[:, :, state_idx]
        state_linear_norms = linear_norms[:, :, state_idx]
        state_tpr_norms = tpr_norms[:, :, state_idx]
        by_state[state_label] = {
            "abs_cosine_similarity": summarize_tensor(state_abs_cosine),
            "linear_norm": summarize_tensor(state_linear_norms),
            "tpr_effective_norm": summarize_tensor(state_tpr_norms),
        }
        abs_cosine_boards[state_label] = state_abs_cosine.tolist()
        linear_norm_boards[state_label] = state_linear_norms.tolist()
        tpr_norm_boards[state_label] = state_tpr_norms.tolist()
        if tpr_bias is not None:
            state_bias = tpr_bias[:, :, state_idx]
            by_state[state_label]["tpr_bias"] = summarize_tensor(state_bias)
            bias_boards[state_label] = state_bias.tolist()

    result = {
        "layer": linear_layer,
        "linear_probe_path": str(linear_probe_path),
        "tpr_probe_path": str(tpr_probe_path),
        "tpr_role_dim": int(tpr_artifact["role_dim"]),
        "tpr_filler_dim": int(tpr_artifact["filler_dim"]),
        "tpr_use_bias": bool(tpr_artifact.get("use_bias", False)),
        "comparison_mode": comparison_mode,
        "comparison_label_kind": comparison_label_kind,
        "comparison_labels": list(comparison_labels),
        "comparison_display_labels": list(comparison_display_labels),
        "mean_center_before_comparison": mean_center_before_comparison,
        "similarity_metric": similarity_metric,
        "overall": {
            "abs_cosine_similarity": summarize_tensor(abs_cosine_similarity),
            "linear_norm": summarize_tensor(linear_norms),
            "tpr_effective_norm": summarize_tensor(tpr_norms),
        },
        "by_state": by_state,
        "abs_cosine_similarity_boards": abs_cosine_boards,
        "linear_norm_boards": linear_norm_boards,
        "tpr_effective_norm_boards": tpr_norm_boards,
        "tpr_bias_boards": bias_boards if tpr_bias is not None else None,
    }
    return result


def print_multilinear_comparison(result: dict, is_no_center: bool = False) -> None:
    overall = result["overall"]["abs_cosine_similarity"]
    comparison_labels = result.get("comparison_labels", list(STATE_LABELS))
    comparison_display_labels = result.get(
        "comparison_display_labels",
        [label.title() for label in comparison_labels],
    )
    print(
        f"Layer {result['layer']}: "
        f"mean |cosine|={overall['mean']:.4f} "
        f"std={overall['std']:.4f} "
        f"min={overall['min']:.4f} "
        f"max={overall['max']:.4f}"
    )
    print(f"  Linear probe:           {result['linear_probe_path']}")
    print(f"  Multilinear TPR probe:  {result['multilinear_probe_path']}")
    if result.get("mean_center_before_comparison", False):
        print(
            "  Mean-centered across the three raw state directions before comparison."
        )
    for state_label, display_label in zip(
        comparison_labels,
        comparison_display_labels,
    ):
        state_summary = result["by_state"][state_label]["abs_cosine_similarity"]
        print(
            f"\n  {display_label} mean |cosine|={state_summary['mean']:.4f} "
            f"std={state_summary['std']:.4f} "
            f"min={state_summary['min']:.4f} "
            f"max={state_summary['max']:.4f}"
        )

    comparison_label_kind = result.get("comparison_label_kind", "state").replace(
        "_", " "
    )


def print_tpr_comparison(result: dict, is_no_center: bool = False) -> None:
    overall = result["overall"]["abs_cosine_similarity"]
    comparison_labels = result.get("comparison_labels", list(STATE_LABELS))
    comparison_display_labels = result.get(
        "comparison_display_labels",
        list(STATE_DISPLAY_LABELS),
    )
    print(
        f"Layer {result['layer']}: "
        f"mean |cosine|={overall['mean']:.4f} "
        f"std={overall['std']:.4f} "
        f"min={overall['min']:.4f} "
        f"max={overall['max']:.4f}"
    )
    print(f"  Linear probe: {result['linear_probe_path']}")
    print(f"  TPR probe:    {result['tpr_probe_path']}")
    if result.get("mean_center_before_comparison", False):
        print(
            "  Mean-centered across the three raw state directions before comparison."
        )
    for state_label, display_label in zip(
        comparison_labels,
        comparison_display_labels,
    ):
        state_summary = result["by_state"][state_label]["abs_cosine_similarity"]
        print(
            f"\n  {display_label} mean |cosine|={state_summary['mean']:.4f} "
            f"std={state_summary['std']:.4f} "
            f"min={state_summary['min']:.4f} "
            f"max={state_summary['max']:.4f}"
        )

    comparison_label_kind = result.get("comparison_label_kind", "state").replace(
        "_", " "
    )


@dataclass
class JointWeightComparisonConfig:
    linear_probe_path: str
    tpr_probe_path: str
    multilinear_tpr_probe_path: str
    device: str = "cpu"
    output_path: str | None = None
    heatmap_output_path: str | None = None
    is_no_center: bool = False
    compare_direction_differences: bool = False
    mean_center_before_comparison: bool = False


def resolve_heatmap_output_path(config: JointWeightComparisonConfig) -> Path:
    if config.heatmap_output_path is not None:
        return Path(config.heatmap_output_path)
    if config.output_path is not None:
        return Path(config.output_path).with_suffix(".pdf")
    stem = "tpr_multilinear_linear"
    if config.mean_center_before_comparison:
        stem += "_mean_centered"
    if config.compare_direction_differences:
        stem += "_direction_difference"
    return Path(f"{stem}_abs_cosine_heatmaps.pdf")


def plot_joint_cosine_heatmaps(
    tpr_result: dict,
    multilinear_result: dict,
    output_path: Path,
    is_no_center: bool,
) -> None:
    comparison_labels = tpr_result.get("comparison_labels", list(STATE_LABELS))
    comparison_display_labels = tpr_result.get(
        "comparison_display_labels",
        [label.title() for label in comparison_labels],
    )
    multilinear_labels = multilinear_result.get("comparison_labels", list(STATE_LABELS))
    if comparison_labels != multilinear_labels:
        raise ValueError(
            "TPR and multilinear comparisons must use the same comparison labels. "
            f"Got {comparison_labels} and {multilinear_labels}."
        )
    if len(comparison_labels) != 3:
        raise ValueError(
            f"Expected exactly three comparison labels for the joint heatmap, got {comparison_labels}"
        )
    method_specs = [
        ("TPR (Bilinear) vs. Linear", tpr_result),
        ("TPR (Trilinear) vs. Linear", multilinear_result),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_paper_style()
    colormap = plt.colormaps["viridis"].copy()
    colormap.set_bad(color="#ececec")

    fig = plt.figure(figsize=(7.5, 5.2))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 3),
        axes_pad=0.18,
        share_all=True,
        cbar_mode="single",
        cbar_location="right",
        cbar_size="4.5%",
        cbar_pad=0.08,
    )
    axes = np.asarray(list(grid), dtype=object).reshape(2, 3)
    cax = grid.cbar_axes[0]
    images = []

    for method_idx, (method_label, result) in enumerate(method_specs):
        for state_idx, (comparison_label, comparison_title) in enumerate(
            zip(comparison_labels, comparison_display_labels)
        ):
            axis = axes[method_idx, state_idx]
            board = apply_display_mask(
                torch.tensor(
                    result["abs_cosine_similarity_boards"][comparison_label],
                    dtype=torch.float32,
                ),
                is_no_center=is_no_center,
            ).numpy()

            image = axis.imshow(
                board,
                cmap=colormap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            images.append(image)

            if method_idx == 0:
                axis.set_title(comparison_title, pad=4.0)
            axis.set_xticks(
                range(BOARD_COLS),
                [str(col_idx) for col_idx in range(1, BOARD_COLS + 1)],
            )
            axis.set_yticks(range(BOARD_ROWS), list(ROW_LABELS))
            if method_idx == 1:
                axis.set_xlabel("Column")
            if state_idx == 0:
                axis.set_ylabel(f"{method_label}\nRow")
            else:
                axis.tick_params(labelleft=False)

            axis.set_xticks(np.arange(-0.5, BOARD_COLS, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, BOARD_ROWS, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.8)
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(which="major", length=0, pad=1.5)
            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#666666")

            for row_idx in range(BOARD_ROWS):
                for col_idx in range(BOARD_COLS):
                    value = board[row_idx, col_idx]
                    if np.isnan(value):
                        label = "N/A"
                        text_color = "#444444"
                    else:
                        label = f"{value:.2f}"
                        rgba = image.cmap(image.norm(value))
                        luminance = (
                            0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                        )
                        text_color = "black" if luminance >= 0.58 else "white"
                    axis.text(
                        col_idx,
                        row_idx,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=text_color,
                    )

    colorbar = cax.colorbar(images[0])
    colorbar.set_label("Absolute cosine similarity")
    colorbar.locator = mticker.MaxNLocator(nbins=5)
    colorbar.update_ticks()
    colorbar.ax.tick_params(length=2.0, width=0.8)
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor("#666666")

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


def compare(config: JointWeightComparisonConfig) -> dict:
    print("Joint weight comparison config:")
    print(json.dumps(asdict(config), indent=2))

    device = torch.device(config.device)
    linear_probe_path = Path(config.linear_probe_path)
    tpr_probe_path = Path(config.tpr_probe_path)
    multilinear_probe_path = Path(config.multilinear_tpr_probe_path)

    tpr_result = compare_tpr_probe_pair(
        linear_probe_path=linear_probe_path,
        tpr_probe_path=tpr_probe_path,
        device=device,
        compare_direction_differences=config.compare_direction_differences,
        mean_center_before_comparison=config.mean_center_before_comparison,
    )
    multilinear_result = compare_multilinear_probe_pair(
        linear_probe_path=linear_probe_path,
        multilinear_probe_path=multilinear_probe_path,
        device=device,
        compare_direction_differences=config.compare_direction_differences,
        mean_center_before_comparison=config.mean_center_before_comparison,
    )

    if int(tpr_result["layer"]) != int(multilinear_result["layer"]):
        raise ValueError(
            "TPR and multilinear checkpoints must come from the same layer for the joint plot. "
            f"Got layer {tpr_result['layer']} and layer {multilinear_result['layer']}."
        )

    print("\nTPR comparison:")
    print_tpr_comparison(tpr_result, is_no_center=config.is_no_center)
    print("Multilinear comparison:")
    print_multilinear_comparison(
        multilinear_result,
        is_no_center=config.is_no_center,
    )

    heatmap_output_path = resolve_heatmap_output_path(config)
    plot_joint_cosine_heatmaps(
        tpr_result=tpr_result,
        multilinear_result=multilinear_result,
        output_path=heatmap_output_path,
        is_no_center=config.is_no_center,
    )
    print(f"Wrote joint absolute-cosine heatmaps to {heatmap_output_path}")

    summary = {
        "config": asdict(config),
        "similarity_metric": tpr_result["similarity_metric"],
        "mean_center_before_comparison": config.mean_center_before_comparison,
        "layer": int(tpr_result["layer"]),
        "heatmap_path": str(heatmap_output_path),
        "tpr_comparison": tpr_result,
        "multilinear_comparison": multilinear_result,
    }

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to {output_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-probe-path", required=True)
    parser.add_argument("--tpr-probe-path", required=True)
    parser.add_argument("--multilinear-tpr-probe-path", required=True)
    parser.add_argument("--device", default=JointWeightComparisonConfig.device)
    parser.add_argument("--output-path")
    parser.add_argument("--heatmap-output-path", required=True)
    parser.add_argument(
        "--is-no-center",
        action="store_true",
        help=(
            "Display the four center squares as N/A in both rows of the heatmap figure "
            "and in the printed board summaries."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = JointWeightComparisonConfig(
        linear_probe_path=args.linear_probe_path,
        tpr_probe_path=args.tpr_probe_path,
        multilinear_tpr_probe_path=args.multilinear_tpr_probe_path,
        device=args.device,
        output_path=args.output_path,
        heatmap_output_path=args.heatmap_output_path,
        is_no_center=args.is_no_center,
        compare_direction_differences=False,
        mean_center_before_comparison=True,
    )
    compare(config)


if __name__ == "__main__":
    main()
