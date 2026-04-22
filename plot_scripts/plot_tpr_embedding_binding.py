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
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from hook_utils import convert_to_hooked_model, record_activations  # noqa: E402
from load_model import load_model  # noqa: E402
from train_probe import (  # noqa: E402
    BLACK,
    WHITE,
    EMPTY,
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


STATE_LABELS = ("empty", "opponent", "current")
ROW_LABELS = "ABCDEFGH"
EMPTY = 0
STATE_COLORS = {
    "empty": "#cbd5e1",
    "opponent": "#2563eb",
    "current": "#d97706",
}


def ensure_hooked_model(model) -> None:
    first_block = next(iter(model.blocks))
    if not hasattr(first_block, "hook_resid_post"):
        convert_to_hooked_model(model)


def square_row_col(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_COLS)


def square_label(index: int) -> str:
    row_idx, col_idx = square_row_col(index)
    return f"{ROW_LABELS[row_idx]}{col_idx + 1}"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_existing_path(raw_path: str | None, *, label: str) -> str | None:
    if raw_path is None:
        return None

    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"{label} path does not exist: {raw_path}")


def resolve_existing_path_from_sources(
    explicit_value: str | None,
    artifact_config: dict,
    key: str,
    fallback: str | None,
    *,
    label: str,
) -> str | None:
    if explicit_value is not None:
        return resolve_existing_path(explicit_value, label=label)

    artifact_value = artifact_config.get(key)
    if artifact_value is not None:
        try:
            return resolve_existing_path(str(artifact_value), label=label)
        except FileNotFoundError:
            pass

    if fallback is not None:
        return resolve_existing_path(fallback, label=label)
    return None


def resolve_from_artifact(
    explicit_value,
    artifact_config: dict,
    key: str,
    fallback,
):
    if explicit_value is not None:
        return explicit_value
    if key in artifact_config:
        return artifact_config[key]
    return fallback


def resolve_index(index: int, length: int, name: str) -> int:
    resolved = index if index >= 0 else length + index
    if not (0 <= resolved < length):
        raise ValueError(f"{name} index {index} is out of range for length {length}")
    return resolved


@dataclass
class BindingEmbeddingPlotConfig:
    probe_path: str
    output_path: str | None = None
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
    label_squares: bool = True
    point_size: float = 72.0
    point_alpha: float = 0.86
    figure_width: float | None = None
    figure_height: float | None = None
    dpi: int = 300
    title: str | None = None


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


def player_name(player: int) -> str:
    return "black" if int(player) == BLACK else "white"


def load_tpr_artifact(
    probe_path: str | Path,
) -> tuple[dict, torch.Tensor, int, int, int]:
    resolved_path = Path(probe_path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"TPR probe checkpoint not found: {resolved_path}")
    artifact = torch.load(resolved_path, map_location="cpu")
    if "probe_state_dict" not in artifact:
        raise KeyError(f"Missing probe_state_dict in {probe_path}")

    state_dict = artifact["probe_state_dict"]
    if "binding_map" not in state_dict:
        raise KeyError(f"Missing binding_map in {probe_path}")

    binding_map = state_dict["binding_map"].detach().cpu().to(torch.float32)
    if binding_map.ndim != 3:
        raise ValueError(
            f"Expected binding_map with shape [d_model, d_r, d_f], got {binding_map.shape}"
        )

    layer = int(artifact["layer"])
    role_dim = int(artifact["role_dim"])
    filler_dim = int(artifact["filler_dim"])
    return artifact, binding_map, layer, role_dim, filler_dim


def plot_groundtruth_board(
    board_state: np.ndarray,
    *,
    title: str,
    ax,
) -> None:
    ax.set_xlim(0, BOARD_COLS)
    ax.set_ylim(BOARD_ROWS, 0)
    ax.set_aspect("equal")
    ax.set_xticks(
        np.arange(BOARD_COLS) + 0.5,
        [str(idx) for idx in range(1, BOARD_COLS + 1)],
    )
    ax.set_yticks(np.arange(BOARD_ROWS) + 0.5, list(ROW_LABELS))
    ax.tick_params(length=0)
    ax.xaxis.tick_top()

    light_square = "#d9f0d0"
    dark_square = "#7fb069"
    for row_idx in range(BOARD_ROWS):
        for col_idx in range(BOARD_COLS):
            square_color = light_square if (row_idx + col_idx) % 2 == 0 else dark_square
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    facecolor=square_color,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )


            board_value = int(board_state[row_idx, col_idx])
            if board_value == EMPTY:
                continue

            piece_face = "#111111" if board_value == BLACK else "#f8fafc"
            piece_edge = "#111111" if board_value in (BLACK, WHITE) else "#666666"
            ax.add_patch(
                Circle(
                    (col_idx + 0.5, row_idx + 0.5),
                    0.34,
                    facecolor=piece_face,
                    edgecolor=piece_edge,
                    linewidth=1.5,
                )
            )

    ax.set_title(title, pad=18)
    ax.set_frame_on(False)


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
    center_squares = np.array(
        sorted(int(square) for square in STARTING_SQUARES), dtype=np.int64
    )
    return square_indices[~np.isin(square_indices, center_squares)]


def compute_pca(
    points: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    if centered.shape[0] == 0:
        raise ValueError("Cannot compute PCA on an empty point set")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    kept_components = min(n_components, vt.shape[0])
    projection = centered @ vt[:kept_components].T
    if projection.shape[1] < n_components:
        projection = np.pad(
            projection,
            ((0, 0), (0, n_components - projection.shape[1])),
            mode="constant",
            constant_values=0.0,
        )

    explained = singular_values**2
    total = float(explained.sum())
    explained_ratio = np.zeros(n_components, dtype=np.float64)
    if total > 0:
        kept = min(n_components, explained.shape[0])
        explained_ratio[:kept] = explained[:kept] / total
    return projection[:, :n_components], explained_ratio


def build_projection(
    points: np.ndarray,
    config: BindingEmbeddingPlotConfig,
) -> tuple[np.ndarray, tuple[str, str]]:
    projection, explained_ratio = compute_pca(
        points,
        n_components=2,
    )
    axis_labels = (
        f"PC1 ({100.0 * explained_ratio[0]:.1f}%)",
        f"PC2 ({100.0 * explained_ratio[1]:.1f}%)",
    )
    return projection, axis_labels


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
    role_embeddings = state_dict["role_embeddings"].to(
        device=device, dtype=torch.float32
    )
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
        square_vectors=square_vectors.detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .reshape(BOARD_ROWS * BOARD_COLS, filler_dim),
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

    metadata = sample.metadata
    title = (
        "TPR Binding Embedding (PCA)\n"
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
    center_tag = "_no_center" if config.exclude_center_squares else ""
    return sample.probe_path.with_name(
        f"{sample.probe_path.stem}_binding_embedding_pca{center_tag}_"
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
    square_projection, filler_projection, axis_labels = (
        project_binding_embedding_points(
            sample,
            config,
        )
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
        "method": "pca",
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
    parser.add_argument(
        "--probe-path", required=True, help="Path to a saved TPR probe."
    )
    parser.add_argument(
        "--output-path",
        default="plots/binding.pdf",
        help="Output image path.",
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
        default=15,
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
    args = parser.parse_args()
    return BindingEmbeddingPlotConfig(
        probe_path=args.probe_path,
        output_path=args.output_path,
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
        label_squares=args.label_squares,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
        title=args.title,
    )


def main() -> None:
    config = parse_args()
    result = run(config)
    print(f"Wrote binding embedding plot to {result['output_path']}")


if __name__ == "__main__":
    main()
