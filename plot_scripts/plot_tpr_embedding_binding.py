"""Project square-level filler vectors induced by a TPR binding tensor."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "hook_utils"))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402

from hook_utils import record_activations  # noqa: E402
from load_model import load_model  # noqa: E402
from plot_tpr_binding_matrix import (  # noqa: E402
    configure_paper_style,
    ensure_hooked_model,
    load_tpr_artifact,
    plot_groundtruth_board,
    resolve_device,
    resolve_existing_path_from_sources,
    resolve_from_artifact,
    resolve_index,
)
from plot_tpr_embedding_isomap import compute_isomap_projection  # noqa: E402
from plot_tpr_embedding_kernel_pca import compute_kernel_pca_projection  # noqa: E402
from plot_tpr_embedding_pca import STATE_LABELS, compute_pca, square_label  # noqa: E402
from plot_tpr_embedding_umap import compute_umap_projection  # noqa: E402
from train_probe import (  # noqa: E402
    BLACK,
    BOARD_COLS,
    BOARD_ROWS,
    OthelloBoardState,
    STARTING_SQUARES,
    load_probe_dataset,
)
from train_tpr_probe import (  # noqa: E402
    compute_tpr_binding_tensor,
    infer_activation_name_from_artifact,
    make_module_name,
)


DEFAULT_METHOD = "pca"
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_METRIC = "euclidean"
DEFAULT_KERNEL = "rbf"
DEFAULT_DEGREE = 3
DEFAULT_COEF0 = 1.0
DEFAULT_EIGEN_SOLVER = "auto"
DEFAULT_RANDOM_SEED = 0
DEFAULT_ISOMAP_N_NEIGHBORS = 16
DEFAULT_ISOMAP_METRIC = "euclidean"

STATE_COLORS = {
    "empty": "#cbd5e1",
    "opponent": "#2563eb",
    "current": "#d97706",
}


@dataclass
class BindingEmbeddingPlotConfig:
    probe_path: str
    output_path: str | None = None
    method: str = DEFAULT_METHOD
    checkpoint: str | None = None
    data_path: str | None = None
    split: str = "valid"
    game_index: int = 0
    position_index: int = 15
    device: str = "auto"
    n_head: int | None = None
    valid_size: int | None = None
    test_size: int | None = None
    seed: int | None = None
    max_games: int | None = None
    exclude_center_squares: bool = False
    standardize: bool = False
    label_squares: bool = True
    point_size: float = 72.0
    point_alpha: float = 0.86
    figure_width: float | None = None
    figure_height: float | None = None
    dpi: int = 300
    title: str | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST
    umap_metric: str = DEFAULT_UMAP_METRIC
    kernel: str = DEFAULT_KERNEL
    gamma: float | None = None
    degree: int = DEFAULT_DEGREE
    coef0: float = DEFAULT_COEF0
    eigen_solver: str = DEFAULT_EIGEN_SOLVER
    isomap_n_neighbors: int = DEFAULT_ISOMAP_N_NEIGHBORS
    isomap_metric: str = DEFAULT_ISOMAP_METRIC


@dataclass
class BindingEmbeddingSample:
    probe_path: Path
    layer: int
    role_dim: int
    filler_dim: int
    square_vectors: np.ndarray
    filler_embeddings: np.ndarray
    board_state: np.ndarray
    relative_state_ids: np.ndarray
    binding_tensor: np.ndarray
    metadata: dict[str, int | str]


def normalize_method_name(raw_method: str) -> str:
    normalized = raw_method.strip().lower().replace("_", "-")
    aliases = {
        "kpca": "kernel-pca",
        "kernelpca": "kernel-pca",
    }
    normalized = aliases.get(normalized, normalized)
    valid_methods = {"pca", "umap", "kernel-pca", "isomap"}
    if normalized not in valid_methods:
        valid_display = ", ".join(sorted(valid_methods))
        raise ValueError(f"Unsupported method {raw_method!r}. Expected one of: {valid_display}")
    return normalized


def player_name(player: int) -> str:
    return "black" if int(player) == BLACK else "white"


def board_state_after_move_prefix(
    raw_game: list[int],
    position_index: int,
) -> tuple[np.ndarray, int]:
    board = OthelloBoardState()
    for move in raw_game[: position_index + 1]:
        board.umpire(int(move))
    return np.array(board.state, copy=True), int(board.next_hand_color)


def board_state_to_relative_state_ids(
    board_state: np.ndarray,
    next_player: int,
) -> np.ndarray:
    relative = np.zeros_like(board_state, dtype=np.int64)
    relative[board_state == -next_player] = 1
    relative[board_state == next_player] = 2
    return relative.reshape(-1)


def selected_square_indices(config: BindingEmbeddingPlotConfig) -> np.ndarray:
    square_indices = np.arange(BOARD_ROWS * BOARD_COLS, dtype=np.int64)
    if not config.exclude_center_squares:
        return square_indices
    center_squares = np.array(sorted(int(square) for square in STARTING_SQUARES), dtype=np.int64)
    return square_indices[~np.isin(square_indices, center_squares)]


def build_projection(
    points: np.ndarray,
    config: BindingEmbeddingPlotConfig,
) -> tuple[np.ndarray, tuple[str, str]]:
    method = normalize_method_name(config.method)
    if method == "pca":
        projection, explained_ratio = compute_pca(
            points,
            n_components=2,
            standardize=config.standardize,
        )
        axis_labels = (
            f"PC1 ({100.0 * explained_ratio[0]:.1f}%)",
            f"PC2 ({100.0 * explained_ratio[1]:.1f}%)",
        )
        return projection, axis_labels

    if method == "umap":
        projection = compute_umap_projection(
            points,
            n_components=2,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            metric=config.umap_metric,
            random_seed=config.random_seed,
            standardize=config.standardize,
        )
        return projection, ("UMAP 1", "UMAP 2")

    if method == "kernel-pca":
        projection = compute_kernel_pca_projection(
            points,
            n_components=2,
            kernel=config.kernel,
            gamma=config.gamma,
            degree=config.degree,
            coef0=config.coef0,
            eigen_solver=config.eigen_solver,
            random_seed=config.random_seed,
            standardize=config.standardize,
        )
        return projection, ("Kernel PC1", "Kernel PC2")

    projection = compute_isomap_projection(
        points,
        n_components=2,
        n_neighbors=config.isomap_n_neighbors,
        metric=config.isomap_metric,
        standardize=config.standardize,
    )
    return projection, ("Isomap 1", "Isomap 2")


def load_binding_embedding_sample(
    config: BindingEmbeddingPlotConfig,
) -> BindingEmbeddingSample:
    probe_path = Path(config.probe_path).expanduser()
    artifact, binding_map, layer, role_dim, filler_dim = load_tpr_artifact(probe_path)
    artifact_config = artifact.get("config", {})

    checkpoint = resolve_existing_path_from_sources(
        config.checkpoint,
        artifact_config,
        "checkpoint",
        "ckpts/synthetic_model.pth",
        label="Model checkpoint",
    )
    data_path = resolve_existing_path_from_sources(
        config.data_path,
        artifact_config,
        "data_path",
        "test_data",
        label="Dataset",
    )
    n_head = int(resolve_from_artifact(config.n_head, artifact_config, "n_head", 8))
    valid_size = int(
        resolve_from_artifact(config.valid_size, artifact_config, "valid_size", 512)
    )
    test_size = int(
        resolve_from_artifact(config.test_size, artifact_config, "test_size", 1000)
    )
    seed = int(resolve_from_artifact(config.seed, artifact_config, "seed", 1111))
    max_games = resolve_from_artifact(
        config.max_games,
        artifact_config,
        "max_games",
        None,
    )

    device = resolve_device(config.device)
    model = load_model(
        {
            "model_path": checkpoint,
            "device": device,
            "n_head": n_head,
        }
    )
    ensure_hooked_model(model)

    split = load_probe_dataset(
        data_path=data_path,
        block_size=model.get_block_size(),
        valid_size=valid_size,
        test_size=test_size,
        seed=seed,
        max_games=max_games,
    )
    if config.split not in ("train", "valid", "test"):
        raise ValueError(f"Unsupported split: {config.split}")

    split_key = f"{config.split}_tokens"
    raw_key = f"{config.split}_raw"
    tokens = split[split_key]
    raw_games = split[raw_key]
    game_index = resolve_index(config.game_index, len(raw_games), "game")

    raw_game = [int(move) for move in raw_games[game_index]]
    batch_tokens = tokens[game_index : game_index + 1].to(device)
    activation_name = infer_activation_name_from_artifact(artifact)
    module_name = artifact.get("module_name", make_module_name(layer, activation_name))

    with torch.inference_mode():
        with record_activations(model, [module_name]) as cache:
            model(batch_tokens[:, :-1])

    residuals = cache[module_name][0][0].to(device=device, dtype=torch.float32)
    position_index = resolve_index(
        config.position_index,
        residuals.shape[0],
        "position",
    )

    residual = residuals[position_index : position_index + 1].unsqueeze(0)
    binding_tensor = compute_tpr_binding_tensor(
        residual,
        binding_map.to(device=device),
    )[0, 0]

    state_dict = artifact["probe_state_dict"]
    role_embeddings = state_dict["role_embeddings"].to(device=device, dtype=torch.float32)
    filler_embeddings = (
        state_dict["filler_embeddings"].detach().cpu().to(torch.float32).numpy()
    )
    if role_embeddings.shape != (BOARD_ROWS, BOARD_COLS, role_dim):
        raise ValueError(
            "Unexpected role embedding shape "
            f"{tuple(role_embeddings.shape)} for role_dim={role_dim}"
        )
    if filler_embeddings.shape != (len(STATE_LABELS), filler_dim):
        raise ValueError(
            "Unexpected filler embedding shape "
            f"{tuple(filler_embeddings.shape)} for filler_dim={filler_dim}"
        )

    square_vectors = torch.einsum(
        "rf,xyr->xyf",
        binding_tensor,
        role_embeddings,
    )
    board_state, next_player = board_state_after_move_prefix(raw_game, position_index)
    relative_state_ids = board_state_to_relative_state_ids(board_state, next_player)
    move = int(raw_game[position_index])
    metadata: dict[str, int | str] = {
        "split": config.split,
        "game_index": game_index,
        "position_index": position_index,
        "ply": position_index + 1,
        "move": move,
        "move_label": square_label(move),
        "next_player": player_name(next_player),
    }

    return BindingEmbeddingSample(
        probe_path=probe_path,
        layer=layer,
        role_dim=role_dim,
        filler_dim=filler_dim,
        square_vectors=square_vectors.detach().cpu().to(torch.float32).numpy().reshape(
            BOARD_ROWS * BOARD_COLS, filler_dim
        ),
        filler_embeddings=filler_embeddings,
        board_state=board_state,
        relative_state_ids=relative_state_ids,
        binding_tensor=binding_tensor.detach().cpu().to(torch.float32).numpy(),
        metadata=metadata,
    )


def build_default_title(
    config: BindingEmbeddingPlotConfig,
    sample: BindingEmbeddingSample,
) -> str:
    if config.title is not None:
        return config.title

    method = normalize_method_name(config.method).replace("-", " ").upper()
    metadata = sample.metadata
    title = (
        f"TPR Binding Embedding ({method})\n"
        f"layer {sample.layer}, r={sample.role_dim}, f={sample.filler_dim}, "
        f"{metadata['split']} game {metadata['game_index']}, "
        f"position {metadata['position_index']}, move {metadata['move_label']}"
    )
    if config.exclude_center_squares:
        title += " [center excluded]"
    return title


def build_default_board_title(sample: BindingEmbeddingSample) -> str:
    metadata = sample.metadata
    return (
        "Ground-Truth Board State\n"
        f"after ply {metadata['ply']}: {metadata['move_label']}, "
        f"next={metadata['next_player']}"
    )


def resolve_default_output_path(
    config: BindingEmbeddingPlotConfig,
    sample: BindingEmbeddingSample,
) -> Path:
    if config.output_path is not None:
        return Path(config.output_path).expanduser()

    metadata = sample.metadata
    method = normalize_method_name(config.method).replace("-", "_")
    center_tag = "_no_center" if config.exclude_center_squares else ""
    return sample.probe_path.with_name(
        f"{sample.probe_path.stem}_binding_embedding_{method}{center_tag}_"
        f"{metadata['split']}_game{metadata['game_index']}_"
        f"pos{metadata['position_index']}.png"
    )


def project_binding_embedding_points(
    sample: BindingEmbeddingSample,
    config: BindingEmbeddingPlotConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    square_indices = selected_square_indices(config)
    all_points = np.concatenate(
        [sample.square_vectors[square_indices], sample.filler_embeddings],
        axis=0,
    )
    projection, axis_labels = build_projection(all_points, config)
    n_square_points = len(square_indices)
    return (
        projection[:n_square_points],
        projection[n_square_points:],
        axis_labels,
    )


def plot_binding_embedding_projection(
    sample: BindingEmbeddingSample,
    *,
    square_projection: np.ndarray,
    filler_projection: np.ndarray,
    axis_labels: tuple[str, str],
    config: BindingEmbeddingPlotConfig,
    output_path: Path | None = None,
):
    square_indices = selected_square_indices(config)
    if len(square_projection) != len(square_indices):
        raise ValueError(
            "square_projection length does not match the selected square set "
            f"({len(square_projection)} vs {len(square_indices)})"
        )
    relative_state_ids = sample.relative_state_ids[square_indices]
    configure_paper_style()
    figure_width = 11.0 if config.figure_width is None else config.figure_width
    figure_height = 6.8 if config.figure_height is None else config.figure_height
    fig, (scatter_ax, board_ax) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3.5, 1.7]},
    )

    for state_id, state_label in enumerate(STATE_LABELS):
        mask = relative_state_ids == state_id
        if not np.any(mask):
            continue
        scatter_ax.scatter(
            square_projection[mask, 0],
            square_projection[mask, 1],
            s=config.point_size,
            alpha=config.point_alpha,
            color=STATE_COLORS[state_label],
            edgecolor="white",
            linewidth=0.7,
            label=f"{state_label} squares",
        )

    if config.label_squares:
        for square_index, point in zip(square_indices.tolist(), square_projection):
            annotation = scatter_ax.annotate(
                square_label(square_index),
                xy=(point[0], point[1]),
                xytext=(0.0, 0.0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=5.4,
                color="#0f172a",
                alpha=0.88,
                zorder=6,
            )
            annotation.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1.4, foreground="white", alpha=0.82),
                    path_effects.Normal(),
                ]
            )

    for state_id, state_label in enumerate(STATE_LABELS):
        point = filler_projection[state_id]
        scatter_ax.scatter(
            point[0],
            point[1],
            s=config.point_size * 2.3,
            marker="*",
            color=STATE_COLORS[state_label],
            edgecolor="#111827",
            linewidth=0.9,
            zorder=5,
            label=f"{state_label} filler",
        )
        scatter_ax.annotate(
            state_label,
            xy=(point[0], point[1]),
            xytext=(6.0, 6.0),
            textcoords="offset points",
            fontsize=8.0,
            weight="bold",
            color="#111827",
        )

    scatter_ax.set_xlabel(axis_labels[0])
    scatter_ax.set_ylabel(axis_labels[1])
    scatter_ax.set_title(build_default_title(config, sample), pad=10.0)
    scatter_ax.grid(alpha=0.18, linewidth=0.8)
    scatter_ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.4, zorder=0)
    scatter_ax.axvline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.4, zorder=0)
    scatter_ax.legend(
        loc="best",
        fontsize=7.6,
        frameon=True,
        framealpha=0.94,
    )

    plot_groundtruth_board(
        sample.board_state,
        title=build_default_board_title(sample),
        ax=board_ax,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=config.dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    return fig


def save_binding_embedding_figure(
    sample: BindingEmbeddingSample,
    *,
    config: BindingEmbeddingPlotConfig,
    output_path: Path,
) -> None:
    square_projection, filler_projection, axis_labels = project_binding_embedding_points(
        sample,
        config,
    )
    fig = plot_binding_embedding_projection(
        sample,
        square_projection=square_projection,
        filler_projection=filler_projection,
        axis_labels=axis_labels,
        config=config,
        output_path=output_path,
    )
    plt.close(fig)


def run(config: BindingEmbeddingPlotConfig) -> dict[str, str | int]:
    sample = load_binding_embedding_sample(config)
    output_path = resolve_default_output_path(config, sample)
    save_binding_embedding_figure(
        sample,
        config=config,
        output_path=output_path,
    )
    result: dict[str, str | int] = {
        "probe_path": str(sample.probe_path),
        "output_path": str(output_path),
        "method": normalize_method_name(config.method),
        "layer": sample.layer,
        "role_dim": sample.role_dim,
        "filler_dim": sample.filler_dim,
    }
    result.update(sample.metadata)
    return result


def parse_args() -> BindingEmbeddingPlotConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Project the square-level filler vectors induced by a TPR binding tensor "
            "for one dataset sample and position. The script hooks the model at the "
            "probe layer, computes the exact binding tensor B_t, unbinds it with the "
            "learned role embeddings, and reduces the resulting 64 square vectors to 2D."
        )
    )
    parser.add_argument("--probe-path", required=True, help="Path to a saved TPR probe.")
    parser.add_argument(
        "--output-path",
        default="plots/binding.pdf",
        help="Output image path.",
    )
    parser.add_argument(
        "--method",
        default="pca",
        help="Dimensionality-reduction method used for the square vectors.",
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Model checkpoint used to compute the binding tensor. If omitted, the "
            "script uses the value stored in the TPR probe config when available."
        ),
    )
    parser.add_argument(
        "--data-path",
        help=(
            "Probe dataset path used to select the sample. If omitted, the script uses "
            "the value stored in the TPR probe config when available."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="valid",
        help="Dataset split used to choose the sample.",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=0,
        help="Game index within the selected split. Negative values count from the end.",
    )
    parser.add_argument(
        "--position-index",
        type=int,
        default=-1,
        help="Position index within the selected game. Negative values count from the end.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, for example 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        help="Attention head count used when loading the model.",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        help="Validation split size for dataset loading.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        help="Test split size for dataset loading.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Dataset split seed.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Optional cap on the number of games loaded before splitting.",
    )
    parser.add_argument(
        "--exclude-center-squares",
        action="store_true",
        help="Exclude the four initial center squares from the projected square set.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize vectors before dimensionality reduction.",
    )
    parser.add_argument(
        "--label-squares",
        dest="label_squares",
        action="store_true",
        help="Annotate each projected square point with its board label.",
    )
    parser.add_argument(
        "--no-label-squares",
        dest="label_squares",
        action="store_false",
        help="Disable square-point annotations.",
    )
    parser.set_defaults(label_squares=BindingEmbeddingPlotConfig.label_squares)
    parser.add_argument(
        "--point-size",
        type=float,
        default=BindingEmbeddingPlotConfig.point_size,
        help="Marker size for projected square points.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=BindingEmbeddingPlotConfig.point_alpha,
        help="Marker opacity for projected square points.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        help="Optional figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        help="Optional figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=BindingEmbeddingPlotConfig.dpi,
        help="Saved image DPI.",
    )
    parser.add_argument(
        "--title",
        help="Optional custom title for the main projection panel.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used by stochastic reducers such as UMAP.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=DEFAULT_UMAP_N_NEIGHBORS,
        help="UMAP neighbor count.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=DEFAULT_UMAP_MIN_DIST,
        help="UMAP min_dist value.",
    )
    parser.add_argument(
        "--umap-metric",
        default=DEFAULT_UMAP_METRIC,
        help="UMAP distance metric.",
    )
    parser.add_argument(
        "--kernel",
        default=DEFAULT_KERNEL,
        help="Kernel PCA kernel name.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Optional Kernel PCA gamma value.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=DEFAULT_DEGREE,
        help="Kernel PCA polynomial degree.",
    )
    parser.add_argument(
        "--coef0",
        type=float,
        default=DEFAULT_COEF0,
        help="Kernel PCA coef0 value.",
    )
    parser.add_argument(
        "--eigen-solver",
        default=DEFAULT_EIGEN_SOLVER,
        help="Kernel PCA eigen solver.",
    )
    parser.add_argument(
        "--isomap-n-neighbors",
        type=int,
        default=DEFAULT_ISOMAP_N_NEIGHBORS,
        help="Isomap neighbor count.",
    )
    parser.add_argument(
        "--isomap-metric",
        default=DEFAULT_ISOMAP_METRIC,
        help="Isomap distance metric.",
    )
    args = parser.parse_args()
    return BindingEmbeddingPlotConfig(
        probe_path=args.probe_path,
        output_path=args.output_path,
        method=args.method,
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        split=args.split,
        game_index=args.game_index,
        position_index=args.position_index,
        device=args.device,
        n_head=args.n_head,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        max_games=args.max_games,
        exclude_center_squares=args.exclude_center_squares,
        standardize=args.standardize,
        label_squares=args.label_squares,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
        title=args.title,
        random_seed=args.random_seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        kernel=args.kernel,
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
        eigen_solver=args.eigen_solver,
        isomap_n_neighbors=args.isomap_n_neighbors,
        isomap_metric=args.isomap_metric,
    )


def main() -> None:
    config = parse_args()
    result = run(config)
    print(f"Wrote binding embedding plot to {result['output_path']}")


if __name__ == "__main__":
    main()
