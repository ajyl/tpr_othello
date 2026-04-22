"""Render a single-row factor comparison figure for three multilinear TPR probes."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BOARD_ROWS = 8
BOARD_COLS = 8
ROW_LABELS = "ABCDEFGH"

PAPER_RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DEFAULT_COMPARISON_GAP_RATIO = 0.02


@dataclass
class MultilinearProbeRecord:
    probe_path: Path
    layer: int
    row_dim: int
    col_dim: int
    color_dim: int
    use_bias: bool
    row_embeddings: np.ndarray
    col_embeddings: np.ndarray
    color_embeddings: np.ndarray
    binding_map: np.ndarray


def configure_paper_style() -> None:
    plt.rcParams.update(PAPER_RC_PARAMS)


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, eps)
    return matrix / safe_norms


def centered_normalized_rows(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return normalize_rows(centered)


def cosine_gram(matrix: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(matrix)
    return normalized @ normalized.T


def row_labels_for_factor(factor_kind: str) -> list[str]:
    if factor_kind == "row":
        return [f"{ROW_LABELS[idx]}" for idx in range(BOARD_ROWS)]
    if factor_kind == "column":
        return [f"{idx + 1}" for idx in range(BOARD_COLS)]
    raise ValueError(f"Unsupported factor kind: {factor_kind}")


def compute_similarity_matrices(
    embeddings: np.ndarray,
    *,
    similarity_metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centered_embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

    if similarity_metric == "cosine":
        raw_matrix = cosine_gram(embeddings)
        centered_rows = centered_normalized_rows(embeddings)
        centered_matrix = centered_rows @ centered_rows.T
        return raw_matrix, centered_matrix

    if similarity_metric == "dot":
        raw_matrix = embeddings @ embeddings.T
        centered_matrix = centered_embeddings @ centered_embeddings.T
        return raw_matrix, centered_matrix

    raise ValueError(
        f"Unsupported similarity metric: {similarity_metric!r}. "
        "Expected 'cosine' or 'dot'."
    )


def resolve_color_limits(
    matrices: list[np.ndarray],
    *,
    similarity_metric: str,
) -> tuple[float, float]:
    if similarity_metric == "cosine":
        return -1.0, 1.0

    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)].reshape(-1) for matrix in matrices]
    )
    if finite_values.size == 0:
        return -1.0, 1.0

    max_abs_value = float(np.max(np.abs(finite_values)))
    if np.isclose(max_abs_value, 0.0):
        max_abs_value = 1.0
    return -max_abs_value, max_abs_value


def resolve_output_path(
    *,
    output_path: Path | None,
    probe_path: Path,
    factor_kind: str,
) -> Path:
    if output_path is not None:
        return output_path
    return probe_path.with_name(
        f"{probe_path.stem}_{factor_kind}_single_factor_comparison_figure.pdf"
    )


def resolve_figure_size(
    figure_width: float | None,
    figure_height: float | None,
) -> tuple[float, float]:
    default_width = 13.2
    default_height = 4
    return (
        default_width if figure_width is None else figure_width,
        default_height if figure_height is None else figure_height,
    )


def load_probe_records(probe_paths: Iterable[Path]) -> list[MultilinearProbeRecord]:
    records: list[MultilinearProbeRecord] = []
    for probe_path in probe_paths:
        artifact = torch.load(probe_path, map_location="cpu")
        state_dict = artifact["probe_state_dict"]
        records.append(
            MultilinearProbeRecord(
                probe_path=probe_path,
                layer=int(artifact["layer"]),
                row_dim=int(artifact["row_dim"]),
                col_dim=int(artifact["col_dim"]),
                color_dim=int(artifact["color_dim"]),
                use_bias=bool(artifact.get("use_bias", False)),
                row_embeddings=(
                    state_dict["row_embeddings"]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                ),
                col_embeddings=(
                    state_dict["col_embeddings"]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                ),
                color_embeddings=(
                    state_dict["color_embeddings"]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                ),
                binding_map=(
                    state_dict["binding_map"]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .reshape(state_dict["binding_map"].shape[0], -1)
                    .numpy()
                ),
            )
        )
    return records


def annotation_text_color(value: float, image) -> str:
    rgba = image.cmap(image.norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance >= 0.58 else "white"


def add_vertical_group_divider(
    fig,
    *,
    left_axis,
    right_axis,
    span_axes,
    color: str = "#666666",
    linewidth: float = 1.0,
) -> None:
    left_bbox = left_axis.get_position()
    right_bbox = right_axis.get_position()
    x_coord = 0.496 * (left_bbox.x1 + right_bbox.x0)
    span_bboxes = [axis.get_position() for axis in span_axes]
    y_bottom = min(bbox.y0 for bbox in span_bboxes) - 0.07
    y_top = max(bbox.y1 for bbox in span_bboxes) + 0.11
    fig.add_artist(
        Line2D(
            [x_coord, x_coord],
            [y_bottom, y_top],
            transform=fig.transFigure,
            color=color,
            linewidth=linewidth,
            alpha=0.95,
        )
    )


def probe_dim_title_suffix(record) -> str:
    return rf"$d_u = {record.row_dim},\ d_v = {record.col_dim}$"


def panel_title(base_title: str, record, *, probe_label: str) -> str:
    return "\n".join([base_title, probe_dim_title_suffix(record)])


def draw_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    annotate: bool,
    annotation_decimals: int,
):
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title, pad=8.0)
    ax.set_xticks(range(len(labels)), labels)  # , rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#666666")

    if annotate:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = float(matrix[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.{annotation_decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=12.0,
                    color=annotation_text_color(value, image),
                )
    return image


def plot_factor_comparison_figure(
    *,
    probe_paths: list[Path],
    factor_kind: str,
    output_path: Path,
    similarity_metric: str,
    cmap: str,
    annotate: bool,
    annotation_decimals: int,
    figure_width: float | None,
    figure_height: float | None,
    comparison_gap_ratio: float,
    title: str | None,
) -> Path:
    records = load_probe_records(probe_paths)
    if len(records) != 2:
        raise ValueError(f"Expected exactly two probe checkpoints, got {len(records)}")

    active_factor_kind = "row" if factor_kind == "row" else "column"
    factor_label = "Row" if active_factor_kind == "row" else "Column"
    labels = row_labels_for_factor(active_factor_kind)
    fig_width, fig_height = resolve_figure_size(
        figure_width=figure_width,
        figure_height=figure_height,
    )

    configure_paper_style()
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[
            1.0,
            1.0,
            comparison_gap_ratio,
            1.0,
        ],
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 3]),
    ]

    def select_embeddings(record):
        return (
            record.row_embeddings
            if active_factor_kind == "row"
            else record.col_embeddings
        )

    probe1_raw, probe1_centered = compute_similarity_matrices(
        select_embeddings(records[0]),
        similarity_metric=similarity_metric,
    )
    _probe2_raw, probe2_centered = compute_similarity_matrices(
        select_embeddings(records[1]),
        similarity_metric=similarity_metric,
    )
    panels = [
        (
            probe1_raw,
            panel_title(f"(a) Gram Matrix", records[0], probe_label="Probe 1"),
        ),
        (
            probe1_centered,
            panel_title(
                f"(b) Mean-Centered Gram Matrix",
                records[0],
                probe_label="Probe 1",
            ),
        ),
        (
            _probe2_raw,
            panel_title(
                f"(c) Gram Matrix",
                records[1],
                probe_label="Probe 2",
            ),
        ),
    ]

    vmin, vmax = resolve_color_limits(
        [matrix for matrix, _title in panels],
        similarity_metric=similarity_metric,
    )

    image = None
    panel_annotation_decimals = [2, 2, 1]
    for panel_idx, (axis, (matrix, panel_title_text)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        image = draw_heatmap(
            axis,
            matrix,
            labels=labels,
            title=panel_title_text,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotate=annotate,
            annotation_decimals=(
                panel_annotation_decimals[panel_idx]
                if panel_idx < len(panel_annotation_decimals)
                else annotation_decimals
            ),
        )

    if title:
        fig.suptitle(title, y=1.02)

    colorbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.9,
        pad=0.02,
        fraction=0.035,
    )
    colorbar.set_label(
        "Cosine similarity" if similarity_metric == "cosine" else "Dot product"
    )
    colorbar.ax.tick_params(length=2.0, width=0.8)
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor("#666666")

    fig.canvas.draw()
    add_vertical_group_divider(
        fig,
        left_axis=axes[1],
        right_axis=axes[2],
        span_axes=axes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-path", type=Path, required=True)
    parser.add_argument("--probe-path2", type=Path, required=True)
    parser.add_argument(
        "--factor",
        choices=("row", "col"),
        required=True,
        help="Which single factor to compare across the three probes.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output figure path. Defaults to "
            "<probe_stem>_<factor>_single_factor_comparison_figure.pdf."
        ),
    )
    parser.add_argument(
        "--cmap",
        default="RdBu",
        help="Matplotlib colormap for the heatmaps.",
    )
    parser.add_argument(
        "--similarity-metric",
        choices=("cosine", "dot"),
        default="cosine",
        help="Similarity score to visualize in the Gram panels.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable per-cell numeric annotations.",
    )
    parser.add_argument(
        "--annotation-decimals",
        type=int,
        default=2,
        help="Number of decimals for heatmap annotations.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=None,
        help="Optional figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=None,
        help="Optional figure height in inches.",
    )
    parser.add_argument(
        "--comparison-gap-ratio",
        type=float,
        default=DEFAULT_COMPARISON_GAP_RATIO,
        help=(
            "Width ratio of the spacer between panels 2 and 3, and between 3 and 4. "
            "Larger values create wider gaps."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom figure title.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.comparison_gap_ratio <= 0.0:
        raise ValueError("--comparison-gap-ratio must be positive")

    probe_paths = [
        args.probe_path.expanduser(),
        args.probe_path2.expanduser(),
    ]
    for probe_path in probe_paths:
        if not probe_path.is_file():
            raise FileNotFoundError(
                f"Multilinear probe checkpoint not found: {probe_path}"
            )

    output_path = resolve_output_path(
        output_path=args.output_path,
        probe_path=probe_paths[0],
        factor_kind=args.factor,
    )
    result_path = plot_factor_comparison_figure(
        probe_paths=probe_paths,
        factor_kind=args.factor,
        output_path=output_path,
        similarity_metric=args.similarity_metric,
        cmap=args.cmap,
        annotate=not args.no_annotate,
        annotation_decimals=args.annotation_decimals,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        comparison_gap_ratio=args.comparison_gap_ratio,
        title=args.title,
    )
    print(f"Wrote single-factor comparison figure to {result_path}")


if __name__ == "__main__":
    main()
