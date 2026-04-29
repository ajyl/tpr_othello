"""Compare two TPR-family probes against their matching linear probes in a shared figure."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from mpl_toolkits.axes_grid1 import ImageGrid  # noqa: E402

BOARD_ROWS = 8
BOARD_COLS = 8
ROW_LABELS = "ABCDEFGH"
STATE_LABELS = ("empty", "opponent", "current")
STATE_DISPLAY_LABELS = ("Empty", "Opponent", "Current")

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


@dataclass
class LocalDistributedWeightComparisonConfig:
    linear_probe_dir: str = "probes/linear"
    local_linear_probe_path: str | None = None
    distributed_linear_probe_path: str | None = None
    local_tpr_probe_path: str | None = None
    distributed_tpr_probe_path: str | None = None
    device: str = "cpu"
    output_path: str | None = None
    heatmap_output_path: str | None = None
    is_no_center: bool = False
    mean_center_before_comparison: bool = False


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


def resolve_comparison_metadata() -> (
    tuple[tuple[str, ...], tuple[str, ...], str, str, str]
):
    return (
        STATE_LABELS,
        STATE_DISPLAY_LABELS,
        "states",
        "state",
        "cosine_similarity",
    )


def compute_cosine_similarity(linear_weights: Tensor, tpr_weights: Tensor) -> Tensor:
    return F.cosine_similarity(
        linear_weights.permute(1, 2, 3, 0),
        tpr_weights.permute(1, 2, 3, 0),
        dim=-1,
        eps=1e-12,
    )


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


def build_display_mask(is_no_center: bool) -> Tensor:
    mask = torch.ones(BOARD_ROWS, BOARD_COLS, dtype=torch.bool)
    if is_no_center:
        for board_pos in STARTING_SQUARES:
            row_idx, col_idx = divmod(board_pos, BOARD_COLS)
            mask[row_idx, col_idx] = False
    return mask


def apply_display_mask(board: Tensor, is_no_center: bool) -> Tensor:
    board = board.detach().cpu().clone()
    if is_no_center:
        mask = build_display_mask(is_no_center=True)
        board[~mask] = float("nan")
    return board


def summarize_tensor(values: Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def compare_tpr_probe_pair(
    linear_probe_path: Path,
    tpr_probe_path: Path,
    device: torch.device,
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

    cosine_similarity = compute_cosine_similarity(linear_weights, tpr_weights)
    linear_norms = linear_weights.norm(dim=0)
    tpr_norms = tpr_weights.norm(dim=0)

    by_state = {}
    cosine_boards = {}
    linear_norm_boards = {}
    tpr_norm_boards = {}
    bias_boards = {}
    for state_idx, state_label in enumerate(comparison_labels):
        state_cosine = cosine_similarity[:, :, state_idx]
        state_linear_norms = linear_norms[:, :, state_idx]
        state_tpr_norms = tpr_norms[:, :, state_idx]
        by_state[state_label] = {
            "cosine_similarity": summarize_tensor(state_cosine),
            "linear_norm": summarize_tensor(state_linear_norms),
            "tpr_effective_norm": summarize_tensor(state_tpr_norms),
        }
        cosine_boards[state_label] = state_cosine.tolist()
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
            "cosine_similarity": summarize_tensor(cosine_similarity),
            "linear_norm": summarize_tensor(linear_norms),
            "tpr_effective_norm": summarize_tensor(tpr_norms),
        },
        "by_state": by_state,
        "cosine_similarity_boards": cosine_boards,
        "linear_norm_boards": linear_norm_boards,
        "tpr_effective_norm_boards": tpr_norm_boards,
        "tpr_bias_boards": bias_boards if tpr_bias is not None else None,
    }
    return result


def mean_center_weights(weights: Tensor) -> Tensor:
    if weights.shape[-1] != len(STATE_LABELS):
        raise ValueError(
            "Mean-centering expects the final dimension to match the three probe "
            f"states, got shape {tuple(weights.shape)}"
        )
    return weights - weights.mean(dim=-1, keepdim=True)


def compare_multilinear_probe_pair(
    linear_probe_path: Path,
    multilinear_probe_path: Path,
    device: torch.device,
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

    cosine_similarity = compute_cosine_similarity(linear_weights, multilinear_weights)
    linear_norms = linear_weights.norm(dim=0)
    multilinear_norms = multilinear_weights.norm(dim=0)

    by_state = {}
    cosine_boards = {}
    linear_norm_boards = {}
    multilinear_norm_boards = {}
    bias_boards = {}
    for state_idx, state_label in enumerate(comparison_labels):
        state_cosine = cosine_similarity[:, :, state_idx]
        state_linear_norms = linear_norms[:, :, state_idx]
        state_multilinear_norms = multilinear_norms[:, :, state_idx]
        by_state[state_label] = {
            "cosine_similarity": summarize_tensor(state_cosine),
            "linear_norm": summarize_tensor(state_linear_norms),
            "multilinear_effective_norm": summarize_tensor(state_multilinear_norms),
        }
        cosine_boards[state_label] = state_cosine.tolist()
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
            "cosine_similarity": summarize_tensor(cosine_similarity),
            "linear_norm": summarize_tensor(linear_norms),
            "multilinear_effective_norm": summarize_tensor(multilinear_norms),
        },
        "by_state": by_state,
        "cosine_similarity_boards": cosine_boards,
        "linear_norm_boards": linear_norm_boards,
        "multilinear_effective_norm_boards": multilinear_norm_boards,
        "multilinear_bias_boards": (
            bias_boards if multilinear_bias is not None else None
        ),
    }
    return result


def resolve_heatmap_output_path(config: LocalDistributedWeightComparisonConfig) -> Path:
    if config.heatmap_output_path is not None:
        return Path(config.heatmap_output_path)
    if config.output_path is not None:
        return Path(config.output_path).with_suffix(".pdf")
    stem = "tpr_local_and_distributed"
    if config.mean_center_before_comparison:
        stem += "_mean_centered"
    return Path(f"{stem}_cosine_heatmaps.pdf")


def resolve_linear_probe_path(
    *,
    explicit_linear_probe_path: str | None,
    linear_probe_dir: str,
    probe_path: Path,
) -> Path:
    if explicit_linear_probe_path is not None:
        return Path(explicit_linear_probe_path)

    artifact = torch.load(probe_path, map_location="cpu")
    layer = int(artifact["layer"])
    linear_probe_path = Path(linear_probe_dir) / f"resid_{layer}_linear.pth"
    if not linear_probe_path.exists():
        raise FileNotFoundError(
            f"Missing linear probe checkpoint for layer {layer}: {linear_probe_path}"
        )
    return linear_probe_path


def detect_probe_family(probe_path: Path) -> str:
    artifact = torch.load(probe_path, map_location="cpu")
    probe_kind = artifact.get("probe_kind")
    state_dict = artifact.get("probe_state_dict")

    if probe_kind in {
        "multilinear_tensor_product",
        "multilinear_tensor_product_baseline",
    }:
        return "multilinear_tpr"
    if probe_kind in {
        "tensor_product",
        "tensor_product_baseline",
    }:
        return "tpr"

    if isinstance(state_dict, dict):
        if all(
            key in state_dict
            for key in ("row_embeddings", "col_embeddings", "color_embeddings")
        ):
            return "multilinear_tpr"
        if all(key in state_dict for key in ("role_embeddings", "filler_embeddings")):
            return "tpr"

    raise ValueError(
        f"Could not infer probe family for {probe_path}. " f"probe_kind={probe_kind!r}"
    )


def compare_probe_against_linear(
    *,
    probe_family: str,
    linear_probe_path: Path,
    probe_path: Path,
    device: torch.device,
    mean_center_before_comparison: bool,
) -> tuple[dict, list[dict[str, float | int | str | None]]]:
    if probe_family == "tpr":
        return compare_tpr_probe_pair(
            linear_probe_path=linear_probe_path,
            tpr_probe_path=probe_path,
            device=device,
            mean_center_before_comparison=mean_center_before_comparison,
        )
    if probe_family == "multilinear_tpr":
        return compare_multilinear_probe_pair(
            linear_probe_path=linear_probe_path,
            multilinear_probe_path=probe_path,
            device=device,
            mean_center_before_comparison=mean_center_before_comparison,
        )
    raise ValueError(f"Unsupported probe family {probe_family!r}")


def panel_method_label(row_name: str, probe_family: str, result: dict) -> str:
    if probe_family == "tpr":
        role_dim = result["tpr_role_dim"]
        filler_dim = result["tpr_filler_dim"]
        return f"TPR (Bilinear) vs. Linear\n" + rf"$d_r={role_dim},\ d_f={filler_dim}$"
    if probe_family == "multilinear_tpr":
        row_dim = result["multilinear_row_dim"]
        col_dim = result["multilinear_col_dim"]
        color_dim = result["multilinear_color_dim"]
        return f"TPR (Trilinear) vs. Linear\n" + rf"$d_u={row_dim}, d_v={col_dim}$"
    raise ValueError(f"Unsupported probe family {probe_family!r}")


def plot_joint_cosine_heatmaps(
    *,
    local_probe_family: str,
    distributed_probe_family: str,
    local_result: dict,
    distributed_result: dict,
    output_path: Path,
    is_no_center: bool,
) -> None:
    comparison_labels = local_result.get("comparison_labels", list(STATE_LABELS))
    comparison_display_labels = local_result.get(
        "comparison_display_labels",
        [label.title() for label in comparison_labels],
    )
    distributed_labels = distributed_result.get("comparison_labels", list(STATE_LABELS))
    if comparison_labels != distributed_labels:
        raise ValueError(
            "Local and distributed TPR comparisons must use the same comparison labels. "
            f"Got {comparison_labels} and {distributed_labels}."
        )
    if len(comparison_labels) != 3:
        raise ValueError(
            f"Expected exactly three comparison labels for the joint heatmap, got {comparison_labels}"
        )

    method_specs = [
        (
            panel_method_label("Local", local_probe_family, local_result),
            local_result,
        ),
        (
            panel_method_label(
                "Distributed", distributed_probe_family, distributed_result
            ),
            distributed_result,
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_paper_style()
    colormap = plt.colormaps["viridis"].copy()
    colormap.set_bad(color="#ececec")

    fig = plt.figure(figsize=(7.6, 5.25))
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
                    result["cosine_similarity_boards"][comparison_label],
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
    colorbar.set_label("Cosine similarity")
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


def compare(config: LocalDistributedWeightComparisonConfig) -> dict:
    print("Local/distributed probe weight comparison config:")
    print(json.dumps(asdict(config), indent=2))

    if config.local_tpr_probe_path is None or config.distributed_tpr_probe_path is None:
        raise ValueError(
            "Both local_tpr_probe_path and distributed_tpr_probe_path are required."
        )

    device = torch.device(config.device)
    local_tpr_probe_path = Path(config.local_tpr_probe_path)
    distributed_tpr_probe_path = Path(config.distributed_tpr_probe_path)
    local_probe_family = detect_probe_family(local_tpr_probe_path)
    distributed_probe_family = detect_probe_family(distributed_tpr_probe_path)
    local_linear_probe_path = resolve_linear_probe_path(
        explicit_linear_probe_path=config.local_linear_probe_path,
        linear_probe_dir=config.linear_probe_dir,
        probe_path=local_tpr_probe_path,
    )
    distributed_linear_probe_path = resolve_linear_probe_path(
        explicit_linear_probe_path=config.distributed_linear_probe_path,
        linear_probe_dir=config.linear_probe_dir,
        probe_path=distributed_tpr_probe_path,
    )

    local_result = compare_probe_against_linear(
        probe_family=local_probe_family,
        linear_probe_path=local_linear_probe_path,
        probe_path=local_tpr_probe_path,
        device=device,
        mean_center_before_comparison=config.mean_center_before_comparison,
    )
    distributed_result = compare_probe_against_linear(
        probe_family=distributed_probe_family,
        linear_probe_path=distributed_linear_probe_path,
        probe_path=distributed_tpr_probe_path,
        device=device,
        mean_center_before_comparison=config.mean_center_before_comparison,
    )

    heatmap_output_path = resolve_heatmap_output_path(config)
    plot_joint_cosine_heatmaps(
        local_probe_family=local_probe_family,
        distributed_probe_family=distributed_probe_family,
        local_result=local_result,
        distributed_result=distributed_result,
        output_path=heatmap_output_path,
        is_no_center=config.is_no_center,
    )
    print(f"Wrote joint cosine heatmaps to {heatmap_output_path}")

    summary = {
        "config": asdict(config),
        "similarity_metric": local_result["similarity_metric"],
        "mean_center_before_comparison": config.mean_center_before_comparison,
        "heatmap_path": str(heatmap_output_path),
        "local_probe_family": local_probe_family,
        "distributed_probe_family": distributed_probe_family,
        "local_comparison": local_result,
        "distributed_comparison": distributed_result,
    }

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to {output_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-probe-dir",
        default=LocalDistributedWeightComparisonConfig.linear_probe_dir,
        help=(
            "Directory containing resid_{layer}_linear.pth checkpoints used when "
            "the per-row linear probe path is not given explicitly."
        ),
    )
    parser.add_argument(
        "--local-linear-probe-path",
        "--linear-probe-path-top",
        help=(
            "Optional explicit linear probe checkpoint for the top/local row. "
            "Defaults to resid_{layer}_linear.pth from --linear-probe-dir."
        ),
    )
    parser.add_argument(
        "--distributed-linear-probe-path",
        "--linear-probe-path-bottom",
        help=(
            "Optional explicit linear probe checkpoint for the bottom/distributed row. "
            "Defaults to resid_{layer}_linear.pth from --linear-probe-dir."
        ),
    )
    parser.add_argument(
        "--local-tpr-probe-path",
        "--tpr-probe-path-top",
        "--tpr-probe-path1",
        required=True,
        dest="local_tpr_probe_path",
        help="Path to the local/top TPR or multilinear TPR probe checkpoint.",
    )
    parser.add_argument(
        "--distributed-tpr-probe-path",
        "--tpr-probe-path-bottom",
        "--tpr-probe-path2",
        required=True,
        dest="distributed_tpr_probe_path",
        help="Path to the distributed/bottom TPR or multilinear TPR probe checkpoint.",
    )
    parser.add_argument(
        "--device", default=LocalDistributedWeightComparisonConfig.device
    )
    parser.add_argument("--output-path")
    parser.add_argument("--heatmap-output-path")
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
    config = LocalDistributedWeightComparisonConfig(
        linear_probe_dir=args.linear_probe_dir,
        local_linear_probe_path=args.local_linear_probe_path,
        distributed_linear_probe_path=args.distributed_linear_probe_path,
        local_tpr_probe_path=args.local_tpr_probe_path,
        distributed_tpr_probe_path=args.distributed_tpr_probe_path,
        device=args.device,
        output_path=args.output_path,
        heatmap_output_path=args.heatmap_output_path,
        is_no_center=False,
        mean_center_before_comparison=True,
    )
    compare(config)


if __name__ == "__main__":
    main()
