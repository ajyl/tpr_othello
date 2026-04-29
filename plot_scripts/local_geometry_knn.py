"""Compare local grid-structure summaries for TPR or linear probe checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from local_geometry_helper_funcs import (  # noqa: E402
    NUM_SQUARES,
    STARTING_SQUARES,
    SUPPORTED_METRICS,
    SUPPORTED_SQUARE_EMBEDDING_SOURCES,
    build_grid_adjacency,
    build_probe_metadata,
    evaluate_k_value,
    load_square_embeddings,
    pairwise_distances,
    preprocess_embeddings,
)


STATE_DISPLAY_LABELS = ("Empty", "Opponent", "Current")
STATE_LABELS = ("empty", "opponent", "current")
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



METRIC_SPECS = (
    ("mean_grid_neighbor_hits", "Neighbor"),
    ("mean_same_row_far_count", "Same\nRow"),
    ("mean_same_col_far_count", "Same\nCol"),
    ("mean_same_diagonal_far_count", "Same\nDiag."),
    ("mean_unrelated_count", "Unrelated"),
)
CHANCE_RETRIEVAL_PERCENT = 11.1


@dataclass
class GridStructureComparisonConfig:
    tpr_probe_path: str
    baseline_probe_path: str
    iid_baseline_probe_path: str | None = None
    output_path: str | None = None
    json_output_path: str | None = None
    linear_probe_state: str | None = None
    tpr_linear_probe_state: str | None = None
    baseline_linear_probe_state: str | None = None
    iid_baseline_linear_probe_state: str | None = None
    tpr_square_embedding_source: str = "auto"
    baseline_square_embedding_source: str = "auto"
    iid_baseline_square_embedding_source: str = "auto"
    metric: str = "euclidean"
    mean_center: bool = False
    standardize: bool = False
    normalize: bool = False
    exclude_center_squares: bool = False
    include_diagonals: bool = False
    k: int = 4
    match_groundtruth_degree: bool = False
    as_percentages: bool = False
    figure_width: float = 4.5
    figure_height: float = 2.1
    show_title: bool = False
    title_fontsize: float = 10.0
    axis_label_fontsize: float = 9.5
    tick_label_fontsize: float = 10
    legend_fontsize: float = 9.5
    value_label_fontsize: float = 9.5


def resolve_plot_output_path(config: GridStructureComparisonConfig) -> Path:
    if config.output_path is not None:
        return Path(config.output_path).expanduser()
    tpr_probe_path = Path(config.tpr_probe_path).expanduser()
    baseline_probe_path = Path(config.baseline_probe_path).expanduser()
    if config.iid_baseline_probe_path is not None:
        iid_baseline_probe_path = Path(config.iid_baseline_probe_path).expanduser()
        return tpr_probe_path.with_name(
            f"{tpr_probe_path.stem}_vs_{baseline_probe_path.stem}_vs_"
            f"{iid_baseline_probe_path.stem}_grid_structure.pdf"
        )
    return tpr_probe_path.with_name(
        f"{tpr_probe_path.stem}_vs_{baseline_probe_path.stem}_grid_structure.pdf"
    )


def resolve_json_output_path(
    config: GridStructureComparisonConfig,
    plot_output_path: Path,
) -> Path:
    if config.json_output_path is not None:
        return Path(config.json_output_path).expanduser()
    return plot_output_path.with_suffix(".json")


def active_square_indices(config: GridStructureComparisonConfig) -> list[int]:
    active_squares = [
        square_idx
        for square_idx in range(NUM_SQUARES)
        if not (config.exclude_center_squares and square_idx in STARTING_SQUARES)
    ]
    if len(active_squares) < 2:
        raise ValueError("Need at least two active squares to compare grid structure")
    return active_squares


def validate_k_configuration(
    *,
    config: GridStructureComparisonConfig,
    active_squares: list[int],
    grid_adjacency: np.ndarray,
) -> None:
    if config.match_groundtruth_degree:
        min_grid_degree = int(grid_adjacency.sum(axis=1).min())
        if min_grid_degree <= 0:
            raise ValueError(
                "Ground-truth degree matching requires every active square to have at least one neighbor"
            )
        return

    max_valid_k = len(active_squares) - 1
    if not 1 <= config.k <= max_valid_k:
        raise ValueError(
            f"k must lie in [1, {max_valid_k}] after square filtering; got {config.k}"
        )


def evaluate_probe_local_structure(
    *,
    probe_path: Path,
    square_embedding_source: str,
    linear_probe_state: str | None,
    config: GridStructureComparisonConfig,
    active_squares: list[int],
    grid_adjacency: np.ndarray,
) -> dict[str, object]:
    square_embeddings, artifact = load_probe_square_embeddings(
        probe_path,
        square_embedding_source=square_embedding_source,
        linear_probe_state=linear_probe_state,
    )
    probe_metadata = build_probe_metadata(artifact)
    if "_selected_state" in artifact:
        state = str(artifact["_selected_state"])
        probe_metadata["selected_state"] = state
        probe_metadata["selected_state_index"] = int(artifact["_selected_state_index"])
        probe_metadata["selected_state_display"] = STATE_DISPLAY_LABELS[
            STATE_LABELS.index(state)
        ]
    module_name = artifact.get("module_name")
    if module_name is not None:
        probe_metadata["module_name"] = str(module_name)
    board_state_distribution_kind = artifact.get("board_state_distribution_kind")
    if board_state_distribution_kind is not None:
        probe_metadata["board_state_distribution_kind"] = str(
            board_state_distribution_kind
        )
    processed_embeddings = preprocess_embeddings(
        square_embeddings[active_squares],
        mean_center=config.mean_center,
        standardize=config.standardize,
        normalize=config.normalize,
    )
    distance_matrix = pairwise_distances(processed_embeddings, config.metric)
    evaluation = evaluate_k_value(
        k=None if config.match_groundtruth_degree else config.k,
        active_squares=active_squares,
        distance_matrix=distance_matrix,
        grid_adjacency=grid_adjacency,
        include_diagonals=config.include_diagonals,
        match_groundtruth_degree=config.match_groundtruth_degree,
        include_per_square=True,
    )
    per_square = evaluation.pop("per_square", [])
    return {
        "probe_path": str(probe_path),
        "probe_metadata": probe_metadata,
        "local_summary": evaluation["local_summary"],
        "local_summary_percentage": summarize_local_summary_percentages(per_square),
        "k": evaluation["k"],
        "k_mode": evaluation["k_mode"],
        "effective_k_summary": evaluation["effective_k_summary"],
        "effective_k_histogram": evaluation["effective_k_histogram"],
    }


def infer_probe_display_name(probe_metadata: dict[str, object], default: str) -> str:
    probe_kind = str(probe_metadata.get("probe_kind", "tensor_product"))
    if probe_kind == "linear":
        state = str(probe_metadata.get("selected_state", "current"))
        state_display = STATE_DISPLAY_LABELS[STATE_LABELS.index(state)]
        layer = probe_metadata.get("layer")
        if layer is None:
            return f"Linear ({state_display})"
        return f"Linear L{layer} ({state_display})"
    distribution_kind = str(probe_metadata.get("board_state_distribution_kind", ""))
    if distribution_kind == "iid_random_squarewise_categorical":
        #return "IID Baseline"
        #return "TPR (Random Encodings)"
        return "TPR (OOD)"
    if probe_kind.endswith("_baseline"):
        #return "TPR Baseline"
        return "TPR (Random Coding)"
    return default


def load_probe_square_embeddings(
    probe_path: Path,
    *,
    square_embedding_source: str,
    linear_probe_state: str | None,
) -> tuple[np.ndarray, dict]:
    artifact = torch.load(probe_path, map_location="cpu")
    probe_tensor = artifact.get("probe")
    if isinstance(probe_tensor, torch.Tensor):
        if square_embedding_source != "auto":
            raise ValueError(
                "Linear probes do not support --*-square-embedding-source; "
                "leave it at `auto`."
            )
        if linear_probe_state is None:
            raise ValueError(
                f"Linear probe checkpoint {probe_path} requires a state selection. "
                "Pass --linear-probe-state or the matching per-probe --*-linear-probe-state flag."
            )
        if probe_tensor.ndim != 5 or probe_tensor.shape[0] != 1:
            raise ValueError(
                "Expected linear probe weights with shape "
                f"(1, activation_dim, 8, 8, {len(STATE_LABELS)}), "
                f"got {tuple(probe_tensor.shape)} in {probe_path}"
            )
        if tuple(probe_tensor.shape[2:]) != (8, 8, len(STATE_LABELS)):
            raise ValueError(
                "Expected linear probe board/state axes with shape "
                f"(8, 8, {len(STATE_LABELS)}), "
                f"got {tuple(probe_tensor.shape[2:])} in {probe_path}"
            )
        state_idx = STATE_LABELS.index(linear_probe_state)
        linear_weights = probe_tensor[0].detach().cpu().to(torch.float32)
        square_embeddings = (
            linear_weights[:, :, :, state_idx]
            .permute(1, 2, 0)
            .contiguous()
            .reshape(NUM_SQUARES, -1)
            .numpy()
        )
        artifact = dict(artifact)
        artifact["probe_kind"] = str(artifact.get("probe_kind", "linear"))
        artifact["_square_embedding_source"] = "linear_probe_weights_state_channel"
        artifact["_square_embedding_dim"] = int(square_embeddings.shape[-1])
        artifact["_selected_state"] = linear_probe_state
        artifact["_selected_state_index"] = state_idx
        return square_embeddings, artifact
    if linear_probe_state is not None:
        raise ValueError(
            f"Received linear-probe state `{linear_probe_state}` for non-linear checkpoint "
            f"{probe_path}. Remove the state flag or pass a linear probe."
        )
    return load_square_embeddings(
        probe_path,
        square_embedding_source=square_embedding_source,
    )


def resolve_linear_probe_state(
    *,
    explicit_state: str | None,
    default_state: str | None,
) -> str | None:
    if explicit_state is not None:
        return explicit_state
    return default_state


def summarize_local_summary_percentages(
    per_square_results: list[dict[str, object]],
) -> dict[str, float]:
    percentage_values = {
        "mean_grid_neighbor_hits": [],
        "mean_same_row_far_count": [],
        "mean_same_col_far_count": [],
        "mean_same_diagonal_far_count": [],
        "mean_unrelated_count": [],
    }
    for square_result in per_square_results:
        effective_k = float(square_result["effective_k"])
        if effective_k <= 0.0:
            continue
        percentage_values["mean_grid_neighbor_hits"].append(
            100.0 * float(square_result["grid_neighbor_hits"]) / effective_k
        )
        percentage_values["mean_same_row_far_count"].append(
            100.0 * float(square_result["same_row_far_count"]) / effective_k
        )
        percentage_values["mean_same_col_far_count"].append(
            100.0 * float(square_result["same_col_far_count"]) / effective_k
        )
        percentage_values["mean_same_diagonal_far_count"].append(
            100.0 * float(square_result["same_diagonal_far_count"]) / effective_k
        )
        percentage_values["mean_unrelated_count"].append(
            100.0 * float(square_result["unrelated_count"]) / effective_k
        )
    return {
        metric_name: float(np.mean(values)) if values else 0.0
        for metric_name, values in percentage_values.items()
    }


def plot_local_summary_comparison(
    *,
    series_summaries: list[tuple[str, dict[str, float]]],
    config: GridStructureComparisonConfig,
    output_path: Path,
) -> None:
    if len(series_summaries) < 2:
        raise ValueError("Expected at least two summary series to compare")

    configure_paper_style()
    figure, axis = plt.subplots(
        figsize=(config.figure_width, config.figure_height),
        constrained_layout=True,
    )

    x_positions = np.arange(len(METRIC_SPECS), dtype=np.float64)
    num_series = len(series_summaries)
    bar_width = min(0.26, 0.78 / num_series)
    offsets = (np.arange(num_series, dtype=np.float64) - 0.5 * (num_series - 1)) * bar_width
    style_specs = (
        ("#0072B2", None),
        ("#D55E00", None),
        ("#009E73", None),
    )
    chance_retrieval_value = (
        CHANCE_RETRIEVAL_PERCENT if config.as_percentages else None
    )

    all_values: list[float] = [1.0]
    if chance_retrieval_value is not None:
        all_values.append(chance_retrieval_value)
    bar_groups = []
    for series_idx, (series_label, summary) in enumerate(series_summaries):
        values = [float(summary[metric_name]) for metric_name, _ in METRIC_SPECS]
        all_values.extend(values)
        color, hatch = style_specs[series_idx % len(style_specs)]
        bar_groups.append(
            axis.bar(
                x_positions + offsets[series_idx],
                values,
                width=bar_width,
                label=series_label,
                color=color,
                edgecolor="#3a3a3a",
                linewidth=0.8,
                hatch=hatch,
            )
        )

    axis.set_xticks(x_positions, [display_label for _, display_label in METRIC_SPECS])
    axis.set_ylabel(
        "% of k-Nearest Neighbors"
        if config.as_percentages
        else "Mean count among selected neighbors",
        fontsize=config.axis_label_fontsize,
    )
    if config.match_groundtruth_degree:
        k_label = "k=true_degree"
    else:
        k_label = f"k={config.k}"
    if config.show_title:
        axis.set_title(
            "Local Grid Structure Summary\n"
            f"{k_label}, metric={config.metric}, "
            f"{'8-neighborhood' if config.include_diagonals else '4-neighborhood'}",
            fontsize=config.title_fontsize,
        )
    if chance_retrieval_value is not None:
        axis.axhline(
            chance_retrieval_value,
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.0,
            label=f"Chance ({chance_retrieval_value:.1f}%)",
        )
    axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    if chance_retrieval_value is not None and len(handles) >= 3:
        order = list(range(1, len(handles))) + [0]
    else:
        order = list(range(len(handles)))
    axis.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        loc="upper center",
        fontsize=config.legend_fontsize,
        handlelength=1.6,
        columnspacing=1.4,
        #ncol=2 if chance_retrieval_value is not None else 1,
        ncol=1,
    )

    ymax = max(all_values)
    axis.set_ylim(0.0, ymax * 1.18)

    for bars in bar_groups:
        for bar in bars:
            height = float(bar.get_height())
            #axis.text(
            #    bar.get_x() + bar.get_width() / 2.0,
            #    height + 0.02 * ymax,
            #    f"{height:.1f}" if config.as_percentages else f"{height:.2f}",
            #    ha="center",
            #    va="bottom",
            #    fontsize=config.value_label_fontsize,
            #)

    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    axis.tick_params(axis="x", pad=3.0, labelsize=config.tick_label_fontsize)
    axis.tick_params(axis="y", pad=2.5, labelsize=config.tick_label_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(figure)


def print_local_summary(label: str, result: dict[str, object]) -> None:
    summary = result["local_summary"]
    percentage_summary = result["local_summary_percentage"]
    print(f"{label}:")
    print(
        "  "
        f"grid_hits={summary['mean_grid_neighbor_hits']:.3f} "
        f"same_row_far={summary['mean_same_row_far_count']:.3f} "
        f"same_col_far={summary['mean_same_col_far_count']:.3f} "
        f"same_diagonal_far={summary['mean_same_diagonal_far_count']:.3f} "
        f"unrelated={summary['mean_unrelated_count']:.3f}"
    )
    print(
        "  "
        f"grid_hits_pct={percentage_summary['mean_grid_neighbor_hits']:.1f} "
        f"same_row_far_pct={percentage_summary['mean_same_row_far_count']:.1f} "
        f"same_col_far_pct={percentage_summary['mean_same_col_far_count']:.1f} "
        f"same_diagonal_far_pct={percentage_summary['mean_same_diagonal_far_count']:.1f} "
        f"unrelated_pct={percentage_summary['mean_unrelated_count']:.1f}"
    )
    print(
        "  "
        f"k_mode={result['k_mode']} "
        f"effective_k_mean={result['effective_k_summary']['mean']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tpr-probe-path",
        required=True,
        help="Path to the first probe checkpoint (TPR or linear).",
    )
    parser.add_argument(
        "--baseline-probe-path",
        required=True,
        help="Path to the second probe checkpoint (TPR or linear).",
    )
    parser.add_argument(
        "--iid-baseline-probe-path",
        help=(
            "Optional third probe checkpoint (TPR or linear) for side-by-side "
            "comparison."
        ),
    )
    parser.add_argument(
        "--output-path",
        help="Optional output figure path. Defaults next to the first probe as a PDF.",
    )
    parser.add_argument(
        "--json-output-path",
        help="Optional JSON output path for the plotted summaries.",
    )
    parser.add_argument(
        "--linear-probe-state",
        choices=STATE_LABELS,
        help=(
            "Default state channel to use for any supplied linear probe checkpoints. "
            "Required when a linear probe is passed unless overridden per probe."
        ),
    )
    parser.add_argument(
        "--tpr-linear-probe-state",
        choices=STATE_LABELS,
        help="Optional per-probe state override when --tpr-probe-path points to a linear probe.",
    )
    parser.add_argument(
        "--baseline-linear-probe-state",
        choices=STATE_LABELS,
        help=(
            "Optional per-probe state override when --baseline-probe-path points to a "
            "linear probe."
        ),
    )
    parser.add_argument(
        "--iid-baseline-linear-probe-state",
        choices=STATE_LABELS,
        help=(
            "Optional per-probe state override when --iid-baseline-probe-path points to a "
            "linear probe."
        ),
    )
    parser.add_argument(
        "--tpr-square-embedding-source",
        choices=SUPPORTED_SQUARE_EMBEDDING_SOURCES,
        default=GridStructureComparisonConfig.tpr_square_embedding_source,
        help="How to derive square embeddings for the regular TPR probe.",
    )
    parser.add_argument(
        "--baseline-square-embedding-source",
        choices=SUPPORTED_SQUARE_EMBEDDING_SOURCES,
        default=GridStructureComparisonConfig.baseline_square_embedding_source,
        help="How to derive square embeddings for the baseline TPR probe.",
    )
    parser.add_argument(
        "--iid-baseline-square-embedding-source",
        choices=SUPPORTED_SQUARE_EMBEDDING_SOURCES,
        default=GridStructureComparisonConfig.iid_baseline_square_embedding_source,
        help="How to derive square embeddings for the iid-board baseline TPR probe.",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default=GridStructureComparisonConfig.metric,
        help="Distance metric used to build the local k-NN graph.",
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
        help="Exclude the four starting center squares from the comparison.",
    )
    parser.add_argument(
        "--include-diagonals",
        action="store_true",
        help="Treat diagonal-adjacent squares as valid local neighbors.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=GridStructureComparisonConfig.k,
        help="Fixed k used for the local k-NN evaluation. Ignored with --match-groundtruth-degree.",
    )
    parser.add_argument(
        "--match-groundtruth-degree",
        action="store_true",
        help="Use each square's true board-neighbor count instead of a fixed k.",
    )
    parser.add_argument(
        "--as-percentages",
        action="store_true",
        help=(
            "Plot the local-summary quantities as percentages of selected neighbors "
            "instead of raw mean counts."
        ),
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=GridStructureComparisonConfig.figure_width,
        help="Figure width in inches for the paper plot.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=GridStructureComparisonConfig.figure_height,
        help="Figure height in inches for the paper plot.",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Include a plot title. Off by default for paper-friendly figures.",
    )
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=GridStructureComparisonConfig.title_fontsize,
        help="Font size for the optional plot title.",
    )
    parser.add_argument(
        "--axis-label-fontsize",
        type=float,
        default=GridStructureComparisonConfig.axis_label_fontsize,
        help="Font size for axis labels.",
    )
    parser.add_argument(
        "--tick-label-fontsize",
        type=float,
        default=GridStructureComparisonConfig.tick_label_fontsize,
        help="Font size for x/y tick labels.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=GridStructureComparisonConfig.legend_fontsize,
        help="Font size for the legend.",
    )
    parser.add_argument(
        "--value-label-fontsize",
        type=float,
        default=GridStructureComparisonConfig.value_label_fontsize,
        help="Font size for the numeric labels above the bars.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = GridStructureComparisonConfig(
        tpr_probe_path=args.tpr_probe_path,
        baseline_probe_path=args.baseline_probe_path,
        iid_baseline_probe_path=args.iid_baseline_probe_path,
        output_path=args.output_path,
        json_output_path=args.json_output_path,
        linear_probe_state=args.linear_probe_state,
        tpr_linear_probe_state=args.tpr_linear_probe_state,
        baseline_linear_probe_state=args.baseline_linear_probe_state,
        iid_baseline_linear_probe_state=args.iid_baseline_linear_probe_state,
        tpr_square_embedding_source=args.tpr_square_embedding_source,
        baseline_square_embedding_source=args.baseline_square_embedding_source,
        iid_baseline_square_embedding_source=args.iid_baseline_square_embedding_source,
        metric=args.metric,
        mean_center=args.mean_center,
        standardize=args.standardize,
        normalize=args.normalize,
        exclude_center_squares=args.exclude_center_squares,
        include_diagonals=args.include_diagonals,
        k=args.k,
        match_groundtruth_degree=args.match_groundtruth_degree,
        as_percentages=args.as_percentages,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        show_title=args.show_title,
        title_fontsize=args.title_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_label_fontsize=args.tick_label_fontsize,
        legend_fontsize=args.legend_fontsize,
        value_label_fontsize=args.value_label_fontsize,
    )

    tpr_probe_path = Path(config.tpr_probe_path).expanduser().resolve()
    baseline_probe_path = Path(config.baseline_probe_path).expanduser().resolve()
    iid_baseline_probe_path = (
        None
        if config.iid_baseline_probe_path is None
        else Path(config.iid_baseline_probe_path).expanduser().resolve()
    )
    if not tpr_probe_path.is_file():
        raise FileNotFoundError(f"Regular TPR probe checkpoint not found: {tpr_probe_path}")
    if not baseline_probe_path.is_file():
        raise FileNotFoundError(f"Baseline TPR probe checkpoint not found: {baseline_probe_path}")
    if iid_baseline_probe_path is not None and not iid_baseline_probe_path.is_file():
        raise FileNotFoundError(
            f"IID baseline TPR probe checkpoint not found: {iid_baseline_probe_path}"
        )

    output_path = resolve_plot_output_path(config)
    json_output_path = resolve_json_output_path(config, output_path)

    active_squares = active_square_indices(config)
    grid_adjacency = build_grid_adjacency(
        active_squares,
        include_diagonals=config.include_diagonals,
    )
    validate_k_configuration(
        config=config,
        active_squares=active_squares,
        grid_adjacency=grid_adjacency,
    )

    print("Grid-structure comparison config:")
    print(json.dumps(asdict(config), indent=2))

    tpr_result = evaluate_probe_local_structure(
        probe_path=tpr_probe_path,
        square_embedding_source=config.tpr_square_embedding_source,
        linear_probe_state=resolve_linear_probe_state(
            explicit_state=config.tpr_linear_probe_state,
            default_state=config.linear_probe_state,
        ),
        config=config,
        active_squares=active_squares,
        grid_adjacency=grid_adjacency,
    )
    baseline_result = evaluate_probe_local_structure(
        probe_path=baseline_probe_path,
        square_embedding_source=config.baseline_square_embedding_source,
        linear_probe_state=resolve_linear_probe_state(
            explicit_state=config.baseline_linear_probe_state,
            default_state=config.linear_probe_state,
        ),
        config=config,
        active_squares=active_squares,
        grid_adjacency=grid_adjacency,
    )
    iid_baseline_result = (
        None
        if iid_baseline_probe_path is None
        else evaluate_probe_local_structure(
            probe_path=iid_baseline_probe_path,
            square_embedding_source=config.iid_baseline_square_embedding_source,
            linear_probe_state=resolve_linear_probe_state(
                explicit_state=config.iid_baseline_linear_probe_state,
                default_state=config.linear_probe_state,
            ),
            config=config,
            active_squares=active_squares,
            grid_adjacency=grid_adjacency,
        )
    )

    tpr_label = infer_probe_display_name(tpr_result["probe_metadata"], "TPR (OthelloGPT)")
    baseline_label = infer_probe_display_name(
        baseline_result["probe_metadata"],
        "TPR Baseline",
    )
    print_local_summary(tpr_label, tpr_result)
    print_local_summary(baseline_label, baseline_result)
    if iid_baseline_result is not None:
        iid_baseline_label = infer_probe_display_name(
            iid_baseline_result["probe_metadata"],
            "IID Baseline",
        )
        print_local_summary(iid_baseline_label, iid_baseline_result)
    else:
        iid_baseline_label = None

    plot_local_summary_comparison(
        series_summaries=[
            (
                tpr_label,
                (
                    tpr_result["local_summary_percentage"]
                    if config.as_percentages
                    else tpr_result["local_summary"]
                ),
            ),
            (
                baseline_label,
                (
                    baseline_result["local_summary_percentage"]
                    if config.as_percentages
                    else baseline_result["local_summary"]
                ),
            ),
            *(
                []
                if iid_baseline_result is None or iid_baseline_label is None
                else [
                    (
                        iid_baseline_label,
                        (
                            iid_baseline_result["local_summary_percentage"]
                            if config.as_percentages
                            else iid_baseline_result["local_summary"]
                        ),
                    )
                ]
            ),
        ],
        config=config,
        output_path=output_path,
    )

    result = {
        "config": asdict(config),
        "plot_value_mode": "percentage" if config.as_percentages else "count",
        "active_square_count": len(active_squares),
        "grid_edge_count": int(np.triu(grid_adjacency, k=1).sum()),
        "tpr_probe": tpr_result,
        "baseline_probe": baseline_result,
        "iid_baseline_probe": iid_baseline_result,
        "compared_metrics": {
            metric_name: {
                "display_label": display_label.replace("\n", " "),
                "tpr_count": float(tpr_result["local_summary"][metric_name]),
                "baseline_count": float(baseline_result["local_summary"][metric_name]),
                "tpr_percentage": float(tpr_result["local_summary_percentage"][metric_name]),
                "baseline_percentage": float(
                    baseline_result["local_summary_percentage"][metric_name]
                ),
                "tpr_plot_value": float(
                    (
                        tpr_result["local_summary_percentage"]
                        if config.as_percentages
                        else tpr_result["local_summary"]
                    )[metric_name]
                ),
                "baseline_plot_value": float(
                    (
                        baseline_result["local_summary_percentage"]
                        if config.as_percentages
                        else baseline_result["local_summary"]
                    )[metric_name]
                ),
                **(
                    {}
                    if iid_baseline_result is None
                    else {
                        "iid_baseline_count": float(
                            iid_baseline_result["local_summary"][metric_name]
                        ),
                        "iid_baseline_percentage": float(
                            iid_baseline_result["local_summary_percentage"][metric_name]
                        ),
                        "iid_baseline_plot_value": float(
                            (
                                iid_baseline_result["local_summary_percentage"]
                                if config.as_percentages
                                else iid_baseline_result["local_summary"]
                            )[metric_name]
                        ),
                    }
                ),
            }
            for metric_name, display_label in METRIC_SPECS
        },
        "plot_path": str(output_path),
    }

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with json_output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Wrote grid-structure comparison plot to {output_path}")
    print(f"Wrote grid-structure comparison summary to {json_output_path}")


if __name__ == "__main__":
    main()
