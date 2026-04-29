"""Evaluate whether square-embedding distances factor through board coordinates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import matplotlib.colors as colors

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from local_geometry_helper_funcs import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    STARTING_SQUARES,
    board_chebyshev_distance,
    board_manhattan_distance,
    build_probe_metadata,
    load_role_embeddings,
    preprocess_embeddings,
    square_label,
    square_row_col,
    summarize_values,
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


SUPPORTED_DISTANCE_METRICS = ("squared_euclidean", "euclidean", "cosine")


@dataclass
class RoleDistanceFactorizationConfig:
    probe_path: str
    baseline_probe_path: str | None = None
    iid_baseline_probe_path: str | None = None
    output_path: str | None = None
    gap_grid_plot_path: str | None = None
    metric: str = "squared_euclidean"
    mean_center: bool = False
    standardize: bool = False
    normalize: bool = False
    exclude_center_squares: bool = False
    include_diagonals: bool = False
    include_pair_records: bool = False
    panel_title_fontsize: float = 10
    axis_label_fontsize: float = 10.5
    tick_label_fontsize: float = 10.25
    annotation_fontsize: float = 7.6
    colorbar_label_fontsize: float = 9.75
    colorbar_tick_fontsize: float = 8.5


def resolve_output_path(
    config: RoleDistanceFactorizationConfig,
    probe_path: Path,
    baseline_probe_path: Path | None = None,
    iid_baseline_probe_path: Path | None = None,
) -> Path:
    if config.output_path is not None:
        return Path(config.output_path).expanduser()
    if baseline_probe_path is not None and iid_baseline_probe_path is not None:
        return probe_path.with_name(
            f"{probe_path.stem}_vs_{baseline_probe_path.stem}_vs_"
            f"{iid_baseline_probe_path.stem}_role_distance_factorization.json"
        )
    if baseline_probe_path is not None:
        return probe_path.with_name(
            f"{probe_path.stem}_vs_{baseline_probe_path.stem}_role_distance_factorization.json"
        )
    if iid_baseline_probe_path is not None:
        return probe_path.with_name(
            f"{probe_path.stem}_vs_{iid_baseline_probe_path.stem}_role_distance_factorization.json"
        )
    return probe_path.with_name(f"{probe_path.stem}_role_distance_factorization.json")


def resolve_gap_grid_plot_path(
    config: RoleDistanceFactorizationConfig,
    output_path: Path,
) -> Path:
    if config.gap_grid_plot_path is not None:
        return Path(config.gap_grid_plot_path).expanduser()
    return output_path.with_name(f"{output_path.stem}_row_col_gap_grid.pdf")


def pairwise_distance_matrix(points: np.ndarray, metric: str) -> np.ndarray:
    if metric in {"squared_euclidean", "euclidean"}:
        squared_norms = np.sum(points * points, axis=1, keepdims=True)
        squared_distances = squared_norms + squared_norms.T - 2.0 * (points @ points.T)
        np.maximum(squared_distances, 0.0, out=squared_distances)
        if metric == "squared_euclidean":
            return squared_distances
        return np.sqrt(squared_distances, out=squared_distances)
    if metric == "cosine":
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = points / norms
        similarities = normalized @ normalized.T
        # np.clip(similarities, -1.0, 1.0, out=similarities)
        # return 1.0 - similarities
        return similarities
    raise ValueError(
        f"Unsupported metric {metric!r}; expected one of {SUPPORTED_DISTANCE_METRICS}"
    )


def board_groundtruth_distance(
    square_a: int,
    square_b: int,
    *,
    include_diagonals: bool,
) -> int:
    if include_diagonals:
        return board_chebyshev_distance(square_a, square_b)
    return board_manhattan_distance(square_a, square_b)


def build_pair_dataset(
    active_squares: list[int],
    distance_matrix: np.ndarray,
    *,
    include_diagonals: bool,
) -> dict[str, np.ndarray]:
    left_square_indices = []
    right_square_indices = []
    left_square_labels = []
    right_square_labels = []
    row_gaps = []
    col_gaps = []
    main_diagonal_gaps = []
    anti_diagonal_gaps = []
    groundtruth_distances = []
    interaction_terms = []
    distances = []

    for left_idx, left_square in enumerate(active_squares):
        left_row, left_col = square_row_col(left_square)
        left_main_diagonal = left_row - left_col
        left_anti_diagonal = left_row + left_col
        for right_idx in range(left_idx + 1, len(active_squares)):
            right_square = active_squares[right_idx]
            right_row, right_col = square_row_col(right_square)
            right_main_diagonal = right_row - right_col
            right_anti_diagonal = right_row + right_col
            row_gap = abs(left_row - right_row)
            col_gap = abs(left_col - right_col)
            main_diagonal_gap = abs(left_main_diagonal - right_main_diagonal)
            anti_diagonal_gap = abs(left_anti_diagonal - right_anti_diagonal)

            left_square_indices.append(int(left_square))
            right_square_indices.append(int(right_square))
            left_square_labels.append(square_label(left_square))
            right_square_labels.append(square_label(right_square))
            row_gaps.append(int(row_gap))
            col_gaps.append(int(col_gap))
            main_diagonal_gaps.append(int(main_diagonal_gap))
            anti_diagonal_gaps.append(int(anti_diagonal_gap))
            groundtruth_distances.append(
                int(
                    board_groundtruth_distance(
                        left_square,
                        right_square,
                        include_diagonals=include_diagonals,
                    )
                )
            )
            interaction_terms.append(int(row_gap * col_gap))
            distances.append(float(distance_matrix[left_idx, right_idx]))

    return {
        "left_square_index": np.asarray(left_square_indices, dtype=np.int64),
        "right_square_index": np.asarray(right_square_indices, dtype=np.int64),
        "left_square_label": np.asarray(left_square_labels, dtype=object),
        "right_square_label": np.asarray(right_square_labels, dtype=object),
        "row_gap": np.asarray(row_gaps, dtype=np.float64),
        "col_gap": np.asarray(col_gaps, dtype=np.float64),
        "main_diagonal_gap": np.asarray(main_diagonal_gaps, dtype=np.float64),
        "anti_diagonal_gap": np.asarray(anti_diagonal_gaps, dtype=np.float64),
        "groundtruth_distance": np.asarray(groundtruth_distances, dtype=np.float64),
        "interaction": np.asarray(interaction_terms, dtype=np.float64),
        "distance": np.asarray(distances, dtype=np.float64),
    }


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    num_parameters: int,
) -> dict[str, float | None]:
    residual = target - prediction
    mse = float(np.mean(residual**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residual)))
    target_mean = float(target.mean())
    centered_target = target - target_mean
    total_sum_squares = float(np.sum(centered_target**2))
    residual_sum_squares = float(np.sum(residual**2))

    if total_sum_squares <= 0.0:
        r2 = 1.0 if residual_sum_squares <= 0.0 else 0.0
    else:
        r2 = 1.0 - residual_sum_squares / total_sum_squares

    if target.size > num_parameters:
        adjusted_r2 = 1.0 - (1.0 - r2) * (target.size - 1) / (
            target.size - num_parameters
        )
    else:
        adjusted_r2 = None

    target_std = float(target.std(ddof=0))
    residual_std = float(residual.std(ddof=0))
    if target_std > 0.0 and float(prediction.std(ddof=0)) > 0.0:
        pearson_r = float(np.corrcoef(target, prediction)[0, 1])
    else:
        pearson_r = None

    return {
        "r2": float(r2),
        "adjusted_r2": None if adjusted_r2 is None else float(adjusted_r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "target_std": target_std,
        "residual_std": residual_std,
        "explained_variance_fraction": (
            0.0 if target_std == 0.0 else 1.0 - (residual_std**2) / (target_std**2)
        ),
        "pearson_r": pearson_r,
    }


def fit_linear_model(
    *,
    target: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    feature_names = list(feature_arrays)
    design_columns = [np.ones_like(target, dtype=np.float64)]
    design_columns.extend(
        np.asarray(feature_arrays[name], dtype=np.float64) for name in feature_names
    )
    design_matrix = np.column_stack(design_columns)
    coefficients, *_ = np.linalg.lstsq(design_matrix, target, rcond=None)
    prediction = design_matrix @ coefficients

    model_summary = {
        "model_kind": "linear_regression",
        "num_parameters": int(design_matrix.shape[1]),
        "coefficients": {
            "intercept": float(coefficients[0]),
            **{
                feature_name: float(coefficient)
                for feature_name, coefficient in zip(feature_names, coefficients[1:])
            },
        },
        **regression_metrics(
            target=target,
            prediction=prediction,
            num_parameters=design_matrix.shape[1],
        ),
    }
    return model_summary, prediction


def fit_gap_table_model(
    *,
    target: np.ndarray,
    first_gaps: np.ndarray,
    second_gaps: np.ndarray,
    first_gap_name: str,
    second_gap_name: str,
    max_first_gap: int,
    max_second_gap: int,
) -> tuple[dict[str, object], np.ndarray]:
    prediction = np.empty_like(target)
    mean_distance_by_gap = np.full(
        (max_first_gap + 1, max_second_gap + 1), np.nan, dtype=np.float64
    )
    std_distance_by_gap = np.full(
        (max_first_gap + 1, max_second_gap + 1), np.nan, dtype=np.float64
    )
    count_by_gap = np.zeros((max_first_gap + 1, max_second_gap + 1), dtype=np.int64)
    gap_records = []

    num_populated_cells = 0
    for first_gap in range(max_first_gap + 1):
        for second_gap in range(max_second_gap + 1):
            mask = (first_gaps == first_gap) & (second_gaps == second_gap)
            if not np.any(mask):
                continue
            values = target[mask]
            cell_mean = float(values.mean())
            cell_std = float(values.std(ddof=0))
            prediction[mask] = cell_mean
            mean_distance_by_gap[first_gap, second_gap] = cell_mean
            std_distance_by_gap[first_gap, second_gap] = cell_std
            count_by_gap[first_gap, second_gap] = int(mask.sum())
            num_populated_cells += 1
            gap_records.append(
                {
                    first_gap_name: first_gap,
                    second_gap_name: second_gap,
                    "count": int(mask.sum()),
                    "mean_distance": cell_mean,
                    "std_distance": cell_std,
                    "min_distance": float(values.min()),
                    "max_distance": float(values.max()),
                }
            )

    model_summary = {
        "model_kind": "gap_lookup_table",
        "num_parameters": num_populated_cells,
        "num_populated_gap_cells": num_populated_cells,
        "first_gap_name": first_gap_name,
        "second_gap_name": second_gap_name,
        "gap_records": gap_records,
        "mean_distance_by_gap": [
            [None if np.isnan(value) else float(value) for value in row_values]
            for row_values in mean_distance_by_gap
        ],
        "std_distance_by_gap": [
            [None if np.isnan(value) else float(value) for value in row_values]
            for row_values in std_distance_by_gap
        ],
        "count_by_gap": count_by_gap.tolist(),
        **regression_metrics(
            target=target,
            prediction=prediction,
            num_parameters=max(1, num_populated_cells),
        ),
    }
    return model_summary, prediction


def _coerce_gap_grid(
    mean_distance_by_gap: list[list[float | None]],
) -> np.ndarray:
    grid = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row_values]
            for row_values in mean_distance_by_gap
        ],
        dtype=np.float64,
    )
    if grid.shape != (BOARD_ROWS, BOARD_COLS):
        raise ValueError(
            "Expected an 8x8 row/column gap grid, " f"got shape {tuple(grid.shape)}"
        )
    return grid


def _build_shared_color_norm(
    grids: Sequence[np.ndarray],
) -> tuple[colors.Normalize | None, dict[str, float] | None]:
    finite_chunks = [
        grid[np.isfinite(grid)] for grid in grids if np.isfinite(grid).any()
    ]
    if not finite_chunks:
        return None, None
    finite_values = np.concatenate(finite_chunks, axis=0)

    value_min = float(finite_values.min())
    value_max = float(finite_values.max())
    if value_min < 0.0 and value_max > 0.0:
        half_range = max(abs(value_min), abs(value_max))
        if half_range == 0.0:
            half_range = 1e-12
        return (
            colors.CenteredNorm(vcenter=0.0, halfrange=half_range),
            {"min": value_min, "max": value_max},
        )
    if value_min == value_max:
        epsilon = 1e-12 if value_min == 0.0 else abs(value_min) * 1e-12
        value_min -= epsilon
        value_max += epsilon
    return (
        colors.Normalize(vmin=value_min, vmax=value_max),
        {"min": value_min, "max": value_max},
    )


def _choose_annotation_text_color(
    value: float,
    *,
    cmap: colors.Colormap,
    norm: colors.Normalize | None,
) -> str:
    if not np.isfinite(value) or norm is None:
        return "#444444"
    rgba = cmap(norm(value))
    luminance = (
        0.2126 * float(rgba[0]) + 0.7152 * float(rgba[1]) + 0.0722 * float(rgba[2])
    )
    return "black" if luminance >= 0.6 else "white"


def _format_gap_annotation(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_row_col_gap_grids(
    *,
    mean_distance_by_gap_grids: Sequence[list[list[float | None]]],
    panel_titles: Sequence[str],
    metric: str,
    output_path: Path,
    panel_title_fontsize: float,
    axis_label_fontsize: float,
    tick_label_fontsize: float,
    annotation_fontsize: float,
    colorbar_label_fontsize: float,
    colorbar_tick_fontsize: float,
) -> dict[str, object]:
    if not mean_distance_by_gap_grids:
        raise ValueError("Expected at least one row/column gap grid to plot")
    if len(mean_distance_by_gap_grids) != len(panel_titles):
        raise ValueError("panel_titles must match the number of supplied gap grids")

    grids = [_coerce_gap_grid(grid) for grid in mean_distance_by_gap_grids]
    shared_norm, shared_value_range = _build_shared_color_norm(grids)
    configure_paper_style()
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="#f0f0f0")

    panel_width_in = 2.53
    colorbar_width_in = 0.28
    figure_height_in = panel_width_in + 0.65
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(grids),
        figsize=(panel_width_in * len(grids) + colorbar_width_in, figure_height_in),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)

    images = []
    for ax, grid, title in zip(axes, grids, panel_titles, strict=True):
        # `grid[row_gap, col_gap]` becomes image[y=col_gap, x=row_gap].
        plotted_grid = np.ma.masked_invalid(grid.T)
        image = ax.imshow(
            plotted_grid,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            aspect="equal",
            norm=shared_norm,
        )
        images.append(image)
        ax.set_xlabel(r"$\Delta j$ (Column Gap)", fontsize=axis_label_fontsize)
        ax.set_title(title, fontsize=panel_title_fontsize)
        ax.set_xticks(range(BOARD_ROWS))
        ax.set_yticks(range(BOARD_COLS))
        ax.set_xlim(-0.5, BOARD_ROWS - 0.5)
        ax.set_ylim(BOARD_COLS - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, BOARD_ROWS, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, BOARD_COLS, 1.0), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", which="major", length=2.2, width=0.6, pad=1.5)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")
        for row_gap in range(BOARD_ROWS):
            for col_gap in range(BOARD_COLS):
                value = float(grid[row_gap, col_gap])
                ax.text(
                    row_gap,
                    col_gap,
                    _format_gap_annotation(value),
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                    color=_choose_annotation_text_color(
                        value,
                        cmap=cmap,
                        norm=shared_norm,
                    ),
                )

    axes[0].set_ylabel(r"$\Delta i$ (Row Gap)", fontsize=axis_label_fontsize)
    colorbar = fig.colorbar(
        images[0],
        ax=axes.tolist(),
        fraction=0.04,
        pad=0.02,
        shrink=0.82,
        aspect=26,
    )
    colorbar.set_label(
        f"Mean {metric.replace('_', ' ')} similarity",
        fontsize=colorbar_label_fontsize,
    )
    colorbar.ax.tick_params(
        length=1.5,
        width=0.5,
        pad=1.0,
        labelsize=colorbar_tick_fontsize,
    )
    colorbar.outline.set_linewidth(0.6)

    fig.savefig(
        output_path, dpi=300, bbox_inches="tight", pad_inches=0.01, facecolor="white"
    )
    plt.close(fig)

    return {
        "path": str(output_path),
        "x_axis_gap_name": "row_gap",
        "y_axis_gap_name": "col_gap",
        "matrix_shape": [int(grids[0].shape[0]), int(grids[0].shape[1])],
        "num_subplots": int(len(grids)),
        "subplot_titles": list(panel_titles),
        "shared_colormap": "RdBu",
        "shared_value_range": shared_value_range,
        "y_axis_order": "top_to_bottom_increasing_gap",
        "paper_style": True,
        "cell_annotations": True,
        "font_sizes": {
            "panel_title": float(panel_title_fontsize),
            "axis_label": float(axis_label_fontsize),
            "tick_label": float(tick_label_fontsize),
            "annotation": float(annotation_fontsize),
            "colorbar_label": float(colorbar_label_fontsize),
            "colorbar_tick": float(colorbar_tick_fontsize),
        },
    }


def fit_three_gap_table_model(
    *,
    target: np.ndarray,
    first_gaps: np.ndarray,
    second_gaps: np.ndarray,
    third_gaps: np.ndarray,
    first_gap_name: str,
    second_gap_name: str,
    third_gap_name: str,
    max_first_gap: int,
    max_second_gap: int,
    max_third_gap: int,
) -> tuple[dict[str, object], np.ndarray]:
    prediction = np.empty_like(target)
    mean_distance_by_gap = np.full(
        (max_first_gap + 1, max_second_gap + 1, max_third_gap + 1),
        np.nan,
        dtype=np.float64,
    )
    std_distance_by_gap = np.full(
        (max_first_gap + 1, max_second_gap + 1, max_third_gap + 1),
        np.nan,
        dtype=np.float64,
    )
    count_by_gap = np.zeros(
        (max_first_gap + 1, max_second_gap + 1, max_third_gap + 1),
        dtype=np.int64,
    )
    gap_records = []

    num_populated_cells = 0
    for first_gap in range(max_first_gap + 1):
        for second_gap in range(max_second_gap + 1):
            for third_gap in range(max_third_gap + 1):
                mask = (
                    (first_gaps == first_gap)
                    & (second_gaps == second_gap)
                    & (third_gaps == third_gap)
                )
                if not np.any(mask):
                    continue
                values = target[mask]
                cell_mean = float(values.mean())
                cell_std = float(values.std(ddof=0))
                prediction[mask] = cell_mean
                mean_distance_by_gap[first_gap, second_gap, third_gap] = cell_mean
                std_distance_by_gap[first_gap, second_gap, third_gap] = cell_std
                count_by_gap[first_gap, second_gap, third_gap] = int(mask.sum())
                num_populated_cells += 1
                gap_records.append(
                    {
                        first_gap_name: first_gap,
                        second_gap_name: second_gap,
                        third_gap_name: third_gap,
                        "count": int(mask.sum()),
                        "mean_distance": cell_mean,
                        "std_distance": cell_std,
                        "min_distance": float(values.min()),
                        "max_distance": float(values.max()),
                    }
                )

    model_summary = {
        "model_kind": "gap_lookup_table",
        "num_parameters": num_populated_cells,
        "num_populated_gap_cells": num_populated_cells,
        "first_gap_name": first_gap_name,
        "second_gap_name": second_gap_name,
        "third_gap_name": third_gap_name,
        "gap_records": gap_records,
        "mean_distance_by_gap": [
            [
                [None if np.isnan(value) else float(value) for value in depth_values]
                for depth_values in row_values
            ]
            for row_values in mean_distance_by_gap
        ],
        "std_distance_by_gap": [
            [
                [None if np.isnan(value) else float(value) for value in depth_values]
                for depth_values in row_values
            ]
            for row_values in std_distance_by_gap
        ],
        "count_by_gap": count_by_gap.tolist(),
        **regression_metrics(
            target=target,
            prediction=prediction,
            num_parameters=max(1, num_populated_cells),
        ),
    }
    return model_summary, prediction


def maybe_build_pair_records(
    *,
    include_pair_records: bool,
    pair_dataset: dict[str, np.ndarray],
    model_predictions: dict[str, np.ndarray],
) -> list[dict[str, object]] | None:
    if not include_pair_records:
        return None

    num_pairs = pair_dataset["distance"].shape[0]
    records = []
    for pair_idx in range(num_pairs):
        record = {
            "left_square_index": int(pair_dataset["left_square_index"][pair_idx]),
            "left_square_label": str(pair_dataset["left_square_label"][pair_idx]),
            "right_square_index": int(pair_dataset["right_square_index"][pair_idx]),
            "right_square_label": str(pair_dataset["right_square_label"][pair_idx]),
            "row_gap": int(pair_dataset["row_gap"][pair_idx]),
            "col_gap": int(pair_dataset["col_gap"][pair_idx]),
            "main_diagonal_gap": int(pair_dataset["main_diagonal_gap"][pair_idx]),
            "anti_diagonal_gap": int(pair_dataset["anti_diagonal_gap"][pair_idx]),
            "groundtruth_distance": int(pair_dataset["groundtruth_distance"][pair_idx]),
            "interaction": float(pair_dataset["interaction"][pair_idx]),
            "distance": float(pair_dataset["distance"][pair_idx]),
            "model_predictions": {
                model_name: float(prediction[pair_idx])
                for model_name, prediction in model_predictions.items()
            },
        }
        records.append(record)
    return records


def infer_probe_panel_title(probe_path: Path, probe_metadata: dict[str, object]) -> str:
    distribution_kind = str(probe_metadata.get("board_state_distribution_kind", ""))
    if distribution_kind == "iid_random_squarewise_categorical":
        # panel_prefix = "IID Baseline"
        panel_prefix = "TPR (OOD): $R^2$=0.03"
        return f"{panel_prefix}"
    probe_kind = str(probe_metadata.get("probe_kind", "tensor_product"))
    if probe_kind.endswith("_baseline"):
        # panel_prefix = "TPR Baseline"
        panel_prefix = "TPR (Random Coding): $R^2$=0.24"
    else:
        # panel_prefix = "TPR Probe"
        panel_prefix = r"TPR (OthelloGPT): $R^2$=0.54"
    # return f"{panel_prefix}\n{probe_path.stem}"
    return f"{panel_prefix}"


def analyze_probe(
    *,
    probe_path: Path,
    config: RoleDistanceFactorizationConfig,
) -> dict[str, object]:
    square_embeddings, artifact = load_role_embeddings(probe_path)
    active_squares = [
        square_idx
        for square_idx in range(BOARD_ROWS * BOARD_COLS)
        if not (config.exclude_center_squares and square_idx in STARTING_SQUARES)
    ]
    if len(active_squares) < 2:
        raise ValueError(
            "Need at least two active squares to evaluate pairwise distances"
        )

    processed_embeddings = preprocess_embeddings(
        square_embeddings[active_squares],
        mean_center=config.mean_center,
        standardize=config.standardize,
        normalize=config.normalize,
    )
    distance_matrix = pairwise_distance_matrix(processed_embeddings, config.metric)
    pair_dataset = build_pair_dataset(
        active_squares,
        distance_matrix,
        include_diagonals=config.include_diagonals,
    )
    target = pair_dataset["distance"]

    model_summaries: dict[str, dict[str, object]] = {}
    model_predictions: dict[str, np.ndarray] = {}

    for model_name, feature_arrays in (
        ("row_only_linear", {"row_gap": pair_dataset["row_gap"]}),
        ("column_only_linear", {"col_gap": pair_dataset["col_gap"]}),
        (
            "main_diagonal_only_linear",
            {"main_diagonal_gap": pair_dataset["main_diagonal_gap"]},
        ),
        (
            "anti_diagonal_only_linear",
            {"anti_diagonal_gap": pair_dataset["anti_diagonal_gap"]},
        ),
        (
            "row_column_additive",
            {
                "row_gap": pair_dataset["row_gap"],
                "col_gap": pair_dataset["col_gap"],
            },
        ),
        (
            "diagonal_additive",
            {
                "main_diagonal_gap": pair_dataset["main_diagonal_gap"],
                "anti_diagonal_gap": pair_dataset["anti_diagonal_gap"],
            },
        ),
        (
            "row_column_interaction",
            {
                "row_gap": pair_dataset["row_gap"],
                "col_gap": pair_dataset["col_gap"],
                "row_col_interaction": pair_dataset["interaction"],
            },
        ),
        (
            "row_column_diagonal_interaction",
            {
                "row_gap": pair_dataset["row_gap"],
                "col_gap": pair_dataset["col_gap"],
                "main_diagonal_gap": pair_dataset["main_diagonal_gap"],
                "row_col_interaction": pair_dataset["interaction"],
                "row_main_diagonal_interaction": (
                    pair_dataset["row_gap"] * pair_dataset["main_diagonal_gap"]
                ),
                "col_main_diagonal_interaction": (
                    pair_dataset["col_gap"] * pair_dataset["main_diagonal_gap"]
                ),
                "row_col_main_diagonal_interaction": (
                    pair_dataset["row_gap"]
                    * pair_dataset["col_gap"]
                    * pair_dataset["main_diagonal_gap"]
                ),
            },
        ),
    ):
        summary, prediction = fit_linear_model(
            target=target,
            feature_arrays=feature_arrays,
        )
        if model_name == "row_column_diagonal_interaction":
            summary["diagonal_gap_name"] = "main_diagonal_gap"
        model_summaries[model_name] = summary
        model_predictions[model_name] = prediction

    gap_table_summary, gap_table_prediction = fit_gap_table_model(
        target=target,
        first_gaps=pair_dataset["row_gap"],
        second_gaps=pair_dataset["col_gap"],
        first_gap_name="row_gap",
        second_gap_name="col_gap",
        max_first_gap=BOARD_ROWS - 1,
        max_second_gap=BOARD_COLS - 1,
    )
    model_summaries["row_col_gap_table"] = gap_table_summary
    model_predictions["row_col_gap_table"] = gap_table_prediction

    row_col_diagonal_gap_table_summary, row_col_diagonal_gap_table_prediction = (
        fit_three_gap_table_model(
            target=target,
            first_gaps=pair_dataset["row_gap"],
            second_gaps=pair_dataset["col_gap"],
            third_gaps=pair_dataset["main_diagonal_gap"],
            first_gap_name="row_gap",
            second_gap_name="col_gap",
            third_gap_name="main_diagonal_gap",
            max_first_gap=BOARD_ROWS - 1,
            max_second_gap=BOARD_COLS - 1,
            max_third_gap=BOARD_ROWS + BOARD_COLS - 2,
        )
    )
    row_col_diagonal_gap_table_summary["diagonal_gap_name"] = "main_diagonal_gap"
    model_summaries["row_col_diagonal_gap_table"] = row_col_diagonal_gap_table_summary
    model_predictions["row_col_diagonal_gap_table"] = (
        row_col_diagonal_gap_table_prediction
    )

    diagonal_gap_table_summary, diagonal_gap_table_prediction = fit_gap_table_model(
        target=target,
        first_gaps=pair_dataset["main_diagonal_gap"],
        second_gaps=pair_dataset["anti_diagonal_gap"],
        first_gap_name="main_diagonal_gap",
        second_gap_name="anti_diagonal_gap",
        max_first_gap=2 * (BOARD_ROWS - 1),
        max_second_gap=2 * (BOARD_COLS - 1),
    )
    model_summaries["diagonal_gap_table"] = diagonal_gap_table_summary
    model_predictions["diagonal_gap_table"] = diagonal_gap_table_prediction

    best_single_coordinate_model_name = max(
        (
            "row_only_linear",
            "column_only_linear",
            "main_diagonal_only_linear",
            "anti_diagonal_only_linear",
        ),
        key=lambda name: model_summaries[name]["r2"],
    )
    best_single_coordinate_r2 = float(
        model_summaries[best_single_coordinate_model_name]["r2"]
    )
    additive_r2 = float(model_summaries["row_column_additive"]["r2"])
    diagonal_additive_r2 = float(model_summaries["diagonal_additive"]["r2"])
    interaction_r2 = float(model_summaries["row_column_interaction"]["r2"])
    row_column_diagonal_interaction_r2 = float(
        model_summaries["row_column_diagonal_interaction"]["r2"]
    )
    gap_table_r2 = float(model_summaries["row_col_gap_table"]["r2"])
    row_col_diagonal_gap_table_r2 = float(
        model_summaries["row_col_diagonal_gap_table"]["r2"]
    )
    diagonal_gap_table_r2 = float(model_summaries["diagonal_gap_table"]["r2"])

    pair_records = maybe_build_pair_records(
        include_pair_records=config.include_pair_records,
        pair_dataset=pair_dataset,
        model_predictions=model_predictions,
    )
    probe_metadata = build_probe_metadata(artifact)
    board_state_distribution_kind = artifact.get("board_state_distribution_kind")
    if board_state_distribution_kind is not None:
        probe_metadata["board_state_distribution_kind"] = str(
            board_state_distribution_kind
        )

    result = {
        "probe_path": str(probe_path),
        "probe_metadata": probe_metadata,
        "panel_title": infer_probe_panel_title(probe_path, probe_metadata),
        "active_square_count": len(active_squares),
        "active_squares": [
            {
                "square_index": int(square_idx),
                "square_label": square_label(square_idx),
            }
            for square_idx in active_squares
        ],
        "pair_count": int(target.size),
        "groundtruth_distance_metric": (
            "chebyshev" if config.include_diagonals else "manhattan"
        ),
        "groundtruth_distance_summary": summarize_values(
            pair_dataset["groundtruth_distance"].tolist()
        ),
        "distance_summary": summarize_values(target.tolist()),
        "models": model_summaries,
        "model_comparisons": {
            "best_single_coordinate_model": best_single_coordinate_model_name,
            "best_single_coordinate_r2": best_single_coordinate_r2,
            "row_column_additive_r2_gain_over_best_single_coordinate": additive_r2
            - best_single_coordinate_r2,
            "diagonal_additive_r2_gain_over_best_single_coordinate": diagonal_additive_r2
            - best_single_coordinate_r2,
            "row_column_interaction_r2_gain_over_additive": interaction_r2
            - additive_r2,
            "row_column_diagonal_interaction_r2_gain_over_row_column_interaction": (
                row_column_diagonal_interaction_r2 - interaction_r2
            ),
            "row_column_diagonal_interaction_r2_gain_over_diagonal_additive": (
                row_column_diagonal_interaction_r2 - diagonal_additive_r2
            ),
            "row_col_gap_table_r2_gain_over_row_column_diagonal_interaction": (
                gap_table_r2 - row_column_diagonal_interaction_r2
            ),
            "row_col_diagonal_gap_table_r2_gain_over_row_col_gap_table": (
                row_col_diagonal_gap_table_r2 - gap_table_r2
            ),
            "row_col_diagonal_gap_table_r2_gain_over_row_column_diagonal_interaction": (
                row_col_diagonal_gap_table_r2 - row_column_diagonal_interaction_r2
            ),
            "diagonal_gap_table_r2_gain_over_diagonal_additive": diagonal_gap_table_r2
            - diagonal_additive_r2,
        },
    }
    if pair_records is not None:
        result["pair_records"] = pair_records
    return result


def print_probe_analysis_summary(result: dict[str, object]) -> None:
    probe_metadata = result["probe_metadata"]
    print("Probe metadata:")
    print(json.dumps(probe_metadata, indent=2))
    print(
        f"Active squares: {result['active_square_count']} | "
        f"pair_count: {result['pair_count']} | "
        f"groundtruth_distance_metric: {result['groundtruth_distance_metric']}"
    )
    if result["groundtruth_distance_summary"] is not None:
        groundtruth_summary = result["groundtruth_distance_summary"]
        print(
            "Ground-truth distance summary: "
            f"mean={groundtruth_summary['mean']:.4f} "
            f"std={groundtruth_summary['std']:.4f} "
            f"min={groundtruth_summary['min']:.4f} "
            f"median={groundtruth_summary['median']:.4f} "
            f"max={groundtruth_summary['max']:.4f}"
        )
    if result["distance_summary"] is not None:
        summary = result["distance_summary"]
        print(
            "Distance summary: "
            f"mean={summary['mean']:.4f} "
            f"std={summary['std']:.4f} "
            f"min={summary['min']:.4f} "
            f"median={summary['median']:.4f} "
            f"max={summary['max']:.4f}"
        )

    print("Model fits:")
    for model_name in (
        "row_only_linear",
        "column_only_linear",
        "main_diagonal_only_linear",
        "anti_diagonal_only_linear",
        "row_column_additive",
        "diagonal_additive",
        "row_column_interaction",
        "row_column_diagonal_interaction",
        "row_col_gap_table",
        "row_col_diagonal_gap_table",
        "diagonal_gap_table",
    ):
        model = result["models"][model_name]
        adjusted_r2_text = (
            "n/a" if model["adjusted_r2"] is None else f"{model['adjusted_r2']:.4f}"
        )
        print(
            f"  {model_name}: "
            f"r2={model['r2']:.4f} "
            f"adj_r2={adjusted_r2_text} "
            f"rmse={model['rmse']:.4f} "
            f"mae={model['mae']:.4f}"
        )
        if "coefficients" in model:
            coefficient_text = ", ".join(
                f"{name}={value:.4f}" for name, value in model["coefficients"].items()
            )
            print(f"    coefficients: {coefficient_text}")
            if model_name == "row_column_diagonal_interaction":
                print(f"    diagonal_gap_name={model['diagonal_gap_name']}")
        if model_name in {
            "row_col_gap_table",
            "row_col_diagonal_gap_table",
            "diagonal_gap_table",
        }:
            print(f"    populated_gap_cells={model['num_populated_gap_cells']}")
            if model_name == "row_col_diagonal_gap_table":
                print(f"    diagonal_gap_name={model['diagonal_gap_name']}")

    comparisons = result["model_comparisons"]
    print("Model comparison diagnostics:")
    print(
        f"  best_single_coordinate_model={comparisons['best_single_coordinate_model']} "
        f"(r2={comparisons['best_single_coordinate_r2']:.4f})"
    )
    print(
        "  row_column_additive_r2_gain_over_best_single_coordinate="
        f"{comparisons['row_column_additive_r2_gain_over_best_single_coordinate']:.4f}"
    )
    print(
        "  diagonal_additive_r2_gain_over_best_single_coordinate="
        f"{comparisons['diagonal_additive_r2_gain_over_best_single_coordinate']:.4f}"
    )
    print(
        "  row_column_interaction_r2_gain_over_additive="
        f"{comparisons['row_column_interaction_r2_gain_over_additive']:.4f}"
    )
    print(
        "  row_column_diagonal_interaction_r2_gain_over_row_column_interaction="
        f"{comparisons['row_column_diagonal_interaction_r2_gain_over_row_column_interaction']:.4f}"
    )
    print(
        "  row_column_diagonal_interaction_r2_gain_over_diagonal_additive="
        f"{comparisons['row_column_diagonal_interaction_r2_gain_over_diagonal_additive']:.4f}"
    )
    print(
        "  row_col_gap_table_r2_gain_over_row_column_diagonal_interaction="
        f"{comparisons['row_col_gap_table_r2_gain_over_row_column_diagonal_interaction']:.4f}"
    )
    print(
        "  row_col_diagonal_gap_table_r2_gain_over_row_col_gap_table="
        f"{comparisons['row_col_diagonal_gap_table_r2_gain_over_row_col_gap_table']:.4f}"
    )
    print(
        "  row_col_diagonal_gap_table_r2_gain_over_row_column_diagonal_interaction="
        f"{comparisons['row_col_diagonal_gap_table_r2_gain_over_row_column_diagonal_interaction']:.4f}"
    )
    print(
        "  diagonal_gap_table_r2_gain_over_diagonal_additive="
        f"{comparisons['diagonal_gap_table_r2_gain_over_diagonal_additive']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-path",
        required=True,
        help="Path to a saved bilinear or trilinear TPR probe.",
    )
    parser.add_argument(
        "--baseline-probe-path",
        help=(
            "Optional second probe checkpoint, typically a TPR baseline from "
            "train_tpr_baseline.py, to compare side-by-side."
        ),
    )
    parser.add_argument(
        "--iid-baseline-probe-path",
        help=(
            "Optional third probe checkpoint for the iid-board TPR baseline "
            "trained with train_tpr_baseline.py --iid-random-board-states."
        ),
    )
    parser.add_argument(
        "--output-path",
        help="Optional JSON output path. Defaults next to the probe checkpoint.",
    )
    parser.add_argument(
        "--gap-grid-plot-path",
        help=(
            "Optional output path for the 8x8 row/column gap heatmap. "
            "Defaults next to the JSON output."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_DISTANCE_METRICS,
        default=RoleDistanceFactorizationConfig.metric,
        help="Distance target to factorize. Default is squared Euclidean distance.",
    )
    parser.add_argument(
        "--mean-center",
        action="store_true",
        help="Subtract the mean square embedding vector before computing distances.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score each embedding dimension before computing distances.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each square embedding before computing distances.",
    )
    parser.add_argument(
        "--exclude-center-squares",
        action="store_true",
        help="Exclude the four starting center squares from the factorization test.",
    )
    parser.add_argument(
        "--include-diagonals",
        action="store_true",
        help=(
            "Treat diagonal-adjacent squares as unit-distance neighbors in the "
            "ground-truth board-distance measure, so diagonal pairs have "
            "`groundtruth_distance=1` instead of 2."
        ),
    )
    parser.add_argument(
        "--include-pair-records",
        action="store_true",
        help="Include per-pair observed distances and model predictions in the JSON output.",
    )
    parser.add_argument(
        "--panel-title-fontsize",
        type=float,
        default=RoleDistanceFactorizationConfig.panel_title_fontsize,
        help="Font size for per-panel titles in the gap heatmap.",
    )
    parser.add_argument(
        "--axis-label-fontsize",
        type=float,
        default=RoleDistanceFactorizationConfig.axis_label_fontsize,
        help="Font size for the x/y axis labels in the gap heatmap.",
    )
    parser.add_argument(
        "--tick-label-fontsize",
        type=float,
        default=RoleDistanceFactorizationConfig.tick_label_fontsize,
        help="Font size for x/y tick labels in the gap heatmap.",
    )
    parser.add_argument(
        "--annotation-fontsize",
        type=float,
        default=RoleDistanceFactorizationConfig.annotation_fontsize,
        help="Font size for the per-cell numeric annotations in the gap heatmap.",
    )
    parser.add_argument(
        "--colorbar-label-fontsize",
        type=float,
        default=RoleDistanceFactorizationConfig.colorbar_label_fontsize,
        help="Font size for the colorbar label in the gap heatmap.",
    )
    parser.add_argument(
        "--colorbar-tick-fontsize",
        type=float,
        default=RoleDistanceFactorizationConfig.colorbar_tick_fontsize,
        help="Font size for the colorbar tick labels in the gap heatmap.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RoleDistanceFactorizationConfig(
        probe_path=args.probe_path,
        baseline_probe_path=args.baseline_probe_path,
        iid_baseline_probe_path=args.iid_baseline_probe_path,
        output_path=args.output_path,
        gap_grid_plot_path=args.gap_grid_plot_path,
        metric=args.metric,
        mean_center=args.mean_center,
        standardize=args.standardize,
        normalize=args.normalize,
        exclude_center_squares=args.exclude_center_squares,
        include_diagonals=args.include_diagonals,
        include_pair_records=args.include_pair_records,
        panel_title_fontsize=args.panel_title_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_label_fontsize=args.tick_label_fontsize,
        annotation_fontsize=args.annotation_fontsize,
        colorbar_label_fontsize=args.colorbar_label_fontsize,
        colorbar_tick_fontsize=args.colorbar_tick_fontsize,
    )

    probe_path = Path(config.probe_path).expanduser().resolve()
    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_path}")
    baseline_probe_path = (
        None
        if config.baseline_probe_path is None
        else Path(config.baseline_probe_path).expanduser().resolve()
    )
    iid_baseline_probe_path = (
        None
        if config.iid_baseline_probe_path is None
        else Path(config.iid_baseline_probe_path).expanduser().resolve()
    )
    if baseline_probe_path is not None and not baseline_probe_path.is_file():
        raise FileNotFoundError(
            f"Baseline probe checkpoint not found: {baseline_probe_path}"
        )
    if iid_baseline_probe_path is not None and not iid_baseline_probe_path.is_file():
        raise FileNotFoundError(
            f"IID baseline probe checkpoint not found: {iid_baseline_probe_path}"
        )
    output_path = resolve_output_path(
        config,
        probe_path,
        baseline_probe_path,
        iid_baseline_probe_path,
    )
    gap_grid_plot_path = resolve_gap_grid_plot_path(config, output_path)

    probe_analysis = analyze_probe(
        probe_path=probe_path,
        config=config,
    )
    baseline_probe_analysis = (
        None
        if baseline_probe_path is None
        else analyze_probe(
            probe_path=baseline_probe_path,
            config=config,
        )
    )
    iid_baseline_probe_analysis = (
        None
        if iid_baseline_probe_path is None
        else analyze_probe(
            probe_path=iid_baseline_probe_path,
            config=config,
        )
    )
    panel_results = [probe_analysis]
    if baseline_probe_analysis is not None:
        panel_results.append(baseline_probe_analysis)
    if iid_baseline_probe_analysis is not None:
        panel_results.append(iid_baseline_probe_analysis)

    row_col_gap_grid_plot = plot_row_col_gap_grids(
        mean_distance_by_gap_grids=[
            panel_result["models"]["row_col_gap_table"]["mean_distance_by_gap"]
            for panel_result in panel_results
        ],
        panel_titles=[panel_result["panel_title"] for panel_result in panel_results],
        metric=config.metric,
        output_path=gap_grid_plot_path,
        panel_title_fontsize=config.panel_title_fontsize,
        axis_label_fontsize=config.axis_label_fontsize,
        tick_label_fontsize=config.tick_label_fontsize,
        annotation_fontsize=config.annotation_fontsize,
        colorbar_label_fontsize=config.colorbar_label_fontsize,
        colorbar_tick_fontsize=config.colorbar_tick_fontsize,
    )
    if baseline_probe_analysis is None and iid_baseline_probe_analysis is None:
        result = {
            "probe_path": probe_analysis["probe_path"],
            "config": asdict(config),
            "probe_metadata": probe_analysis["probe_metadata"],
            "active_square_count": probe_analysis["active_square_count"],
            "active_squares": probe_analysis["active_squares"],
            "pair_count": probe_analysis["pair_count"],
            "groundtruth_distance_metric": probe_analysis[
                "groundtruth_distance_metric"
            ],
            "groundtruth_distance_summary": probe_analysis[
                "groundtruth_distance_summary"
            ],
            "distance_summary": probe_analysis["distance_summary"],
            "row_col_gap_grid_plot": row_col_gap_grid_plot,
            "models": probe_analysis["models"],
            "model_comparisons": probe_analysis["model_comparisons"],
        }
        if "pair_records" in probe_analysis:
            result["pair_records"] = probe_analysis["pair_records"]
    else:
        result = {
            "config": asdict(config),
            "comparison_mode": True,
            "probe_path": str(probe_path),
            "baseline_probe_path": (
                None if baseline_probe_path is None else str(baseline_probe_path)
            ),
            "iid_baseline_probe_path": (
                None
                if iid_baseline_probe_path is None
                else str(iid_baseline_probe_path)
            ),
            "probe_analysis": probe_analysis,
            "baseline_probe_analysis": baseline_probe_analysis,
            "iid_baseline_probe_analysis": iid_baseline_probe_analysis,
            "row_col_gap_grid_plot": row_col_gap_grid_plot,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("Role distance-factorization config:")
    print(json.dumps(asdict(config), indent=2))
    print(
        f"metric: {config.metric} | "
        f"groundtruth_distance_metric: {probe_analysis['groundtruth_distance_metric']} | "
        f"local-neighbor mode: {'8-neighborhood' if config.include_diagonals else '4-neighborhood'}"
    )
    print()
    print(f"Analysis for {probe_analysis['panel_title']}:")
    print_probe_analysis_summary(probe_analysis)
    if baseline_probe_analysis is not None:
        print()
        print(f"Analysis for {baseline_probe_analysis['panel_title']}:")
        print_probe_analysis_summary(baseline_probe_analysis)
    if iid_baseline_probe_analysis is not None:
        print()
        print(f"Analysis for {iid_baseline_probe_analysis['panel_title']}:")
        print_probe_analysis_summary(iid_baseline_probe_analysis)
    print(f"Wrote distance-factorization analysis to {output_path}")
    print(f"Wrote row/column gap heatmap to {gap_grid_plot_path}")


if __name__ == "__main__":
    main()
