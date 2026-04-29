"""Evaluate local grid geometry in TPR square embeddings."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent

BOARD_ROWS = 8
BOARD_COLS = 8
NUM_SQUARES = BOARD_ROWS * BOARD_COLS
ROW_LABELS = "ABCDEFGH"
STARTING_SQUARES = frozenset({27, 28, 35, 36})
DEFAULT_K_VALUES = (1, 2, 3, 4, 5, 6, 8, 10)
SUPPORTED_METRICS = ("euclidean", "cosine")
SUPPORTED_GEODESIC_EDGE_METRICS = ("same_as_metric", "euclidean", "cosine", "mahalanobis")
SUPPORTED_SQUARE_EMBEDDING_SOURCES = (
    "auto",
    "role_embeddings",
    "row_col_outer_product",
)
DIRECT_ROLE_EMBEDDING_PROBE_KINDS = {
    "tensor_product",
    "tensor_product_baseline",
}
ROW_COL_OUTER_PRODUCT_PROBE_KINDS = {
    "multilinear_tensor_product",
    "multilinear_tensor_product_baseline",
}


@dataclass
class RoleGridGraphConfig:
    probe_path: str
    output_path: str | None = None
    square_embedding_source: str = "auto"
    metric: str = "euclidean"
    geodesic_edge_metric: str = "mahalanobis"
    mean_center: bool = False
    standardize: bool = False
    normalize: bool = False
    exclude_center_squares: bool = False
    include_diagonals: bool = False
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    match_groundtruth_degree: bool = False
    include_per_square: bool = False


def square_row_col(square_idx: int) -> tuple[int, int]:
    return divmod(square_idx, BOARD_COLS)


def square_label(square_idx: int) -> str:
    row_idx, col_idx = square_row_col(square_idx)
    return f"{ROW_LABELS[row_idx]}{col_idx + 1}"


def board_manhattan_distance(square_a: int, square_b: int) -> int:
    row_a, col_a = square_row_col(square_a)
    row_b, col_b = square_row_col(square_b)
    return abs(row_a - row_b) + abs(col_a - col_b)


def board_chebyshev_distance(square_a: int, square_b: int) -> int:
    row_a, col_a = square_row_col(square_a)
    row_b, col_b = square_row_col(square_b)
    return max(abs(row_a - row_b), abs(col_a - col_b))


def is_local_neighbor(
    square_a: int,
    square_b: int,
    *,
    include_diagonals: bool,
) -> bool:
    if square_a == square_b:
        return False
    if include_diagonals:
        return board_chebyshev_distance(square_a, square_b) == 1
    return board_manhattan_distance(square_a, square_b) == 1


def classify_square_pair(
    square_a: int,
    square_b: int,
    *,
    include_diagonals: bool,
) -> str:
    row_a, col_a = square_row_col(square_a)
    row_b, col_b = square_row_col(square_b)
    row_distance = abs(row_a - row_b)
    col_distance = abs(col_a - col_b)
    if is_local_neighbor(
        square_a,
        square_b,
        include_diagonals=include_diagonals,
    ):
        return "grid_neighbor"
    if row_a == row_b and col_distance > 1:
        return "same_row_far"
    if col_a == col_b and row_distance > 1:
        return "same_col_far"
    if row_distance == col_distance and row_distance > 1:
        return "same_diagonal_far"
    return "unrelated"


def resolve_output_path(config: RoleGridGraphConfig, probe_path: Path) -> Path:
    if config.output_path is not None:
        return Path(config.output_path).expanduser()
    return probe_path.with_name(f"{probe_path.stem}_role_grid_graph.json")


def _load_tensor_from_state_dict(
    state_dict: dict,
    key: str,
    probe_path: Path,
) -> np.ndarray:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"Missing tensor `{key}` in {probe_path}")
    return value.detach().cpu().to(torch.float32).numpy()


def _load_direct_square_embeddings(
    state_dict: dict,
    probe_path: Path,
) -> np.ndarray:
    role_embeddings = _load_tensor_from_state_dict(
        state_dict,
        "role_embeddings",
        probe_path,
    )
    expected_shape = (BOARD_ROWS, BOARD_COLS, role_embeddings.shape[-1])
    if role_embeddings.shape != expected_shape:
        raise ValueError(
            "Expected role embeddings with shape "
            f"({BOARD_ROWS}, {BOARD_COLS}, role_dim), got {tuple(role_embeddings.shape)}"
        )
    return role_embeddings.reshape(NUM_SQUARES, -1)


def _load_row_col_outer_product_square_embeddings(
    state_dict: dict,
    probe_path: Path,
) -> np.ndarray:
    row_embeddings = _load_tensor_from_state_dict(
        state_dict,
        "row_embeddings",
        probe_path,
    )
    col_embeddings = _load_tensor_from_state_dict(
        state_dict,
        "col_embeddings",
        probe_path,
    )
    expected_row_shape = (BOARD_ROWS, row_embeddings.shape[-1])
    expected_col_shape = (BOARD_COLS, col_embeddings.shape[-1])
    if row_embeddings.shape != expected_row_shape:
        raise ValueError(
            "Expected row embeddings with shape "
            f"({BOARD_ROWS}, row_dim), got {tuple(row_embeddings.shape)}"
        )
    if col_embeddings.shape != expected_col_shape:
        raise ValueError(
            "Expected column embeddings with shape "
            f"({BOARD_COLS}, col_dim), got {tuple(col_embeddings.shape)}"
        )
    square_embeddings = np.einsum(
        "ir,jc->ijrc",
        row_embeddings,
        col_embeddings,
    )
    return square_embeddings.reshape(NUM_SQUARES, -1)


def load_square_embeddings(
    probe_path: Path,
    *,
    square_embedding_source: str = "auto",
) -> tuple[np.ndarray, dict]:
    artifact = torch.load(probe_path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected checkpoint dict in {probe_path}, got {type(artifact)}")

    if square_embedding_source not in SUPPORTED_SQUARE_EMBEDDING_SOURCES:
        raise ValueError(
            "Unsupported square embedding source "
            f"{square_embedding_source!r}; expected one of {SUPPORTED_SQUARE_EMBEDDING_SOURCES}"
        )

    probe_kind = artifact.get("probe_kind")
    state_dict = artifact.get("probe_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Missing `probe_state_dict` in {probe_path}")

    has_role_embeddings = isinstance(state_dict.get("role_embeddings"), torch.Tensor)
    has_row_col_embeddings = isinstance(state_dict.get("row_embeddings"), torch.Tensor) and isinstance(
        state_dict.get("col_embeddings"),
        torch.Tensor,
    )

    resolved_source = square_embedding_source
    if resolved_source == "auto":
        if probe_kind in ROW_COL_OUTER_PRODUCT_PROBE_KINDS:
            resolved_source = "row_col_outer_product"
        elif probe_kind in DIRECT_ROLE_EMBEDDING_PROBE_KINDS:
            resolved_source = "role_embeddings"
        elif has_role_embeddings:
            resolved_source = "role_embeddings"
        elif has_row_col_embeddings:
            resolved_source = "row_col_outer_product"
        else:
            raise ValueError(
                "Could not infer how to build square embeddings from checkpoint "
                f"{probe_path}; probe_kind={probe_kind!r}, "
                f"available tensors={sorted(state_dict)}"
            )

    if resolved_source == "role_embeddings":
        if not has_role_embeddings:
            raise ValueError(
                f"Checkpoint {probe_path} does not contain `role_embeddings`, "
                "so `--square-embedding-source role_embeddings` is invalid"
            )
        square_embeddings = _load_direct_square_embeddings(state_dict, probe_path)
    elif resolved_source == "row_col_outer_product":
        if not has_row_col_embeddings:
            raise ValueError(
                f"Checkpoint {probe_path} does not contain both `row_embeddings` and "
                "`col_embeddings`, so `--square-embedding-source row_col_outer_product` "
                "is invalid"
            )
        square_embeddings = _load_row_col_outer_product_square_embeddings(
            state_dict,
            probe_path,
        )
    else:
        raise ValueError(
            "Unsupported resolved square embedding source "
            f"{resolved_source!r}; expected one of {SUPPORTED_SQUARE_EMBEDDING_SOURCES}"
        )

    artifact = dict(artifact)
    artifact["_square_embedding_source"] = resolved_source
    artifact["_square_embedding_dim"] = int(square_embeddings.shape[-1])
    return square_embeddings, artifact


def load_role_embeddings(
    probe_path: Path,
    *,
    square_embedding_source: str = "auto",
) -> tuple[np.ndarray, dict]:
    return load_square_embeddings(
        probe_path,
        square_embedding_source=square_embedding_source,
    )


def _maybe_int(value: object) -> int | None:
    return None if value is None else int(value)


def build_probe_metadata(artifact: dict) -> dict[str, object]:
    metadata: dict[str, object] = {
        "probe_kind": artifact.get("probe_kind", "tensor_product"),
        "use_bias": bool(artifact.get("use_bias", False)),
        "square_embedding_source": artifact.get("_square_embedding_source"),
        "square_embedding_dim": _maybe_int(artifact.get("_square_embedding_dim")),
    }
    for key in (
        "layer",
        "role_dim",
        "filler_dim",
        "row_dim",
        "col_dim",
        "color_dim",
    ):
        maybe_value = _maybe_int(artifact.get(key))
        if maybe_value is not None:
            metadata[key] = maybe_value
    return metadata


def preprocess_embeddings(
    embeddings: np.ndarray,
    *,
    mean_center: bool,
    standardize: bool,
    normalize: bool,
) -> np.ndarray:
    processed = np.asarray(embeddings, dtype=np.float64)
    if mean_center:
        processed = processed - processed.mean(axis=0, keepdims=True)
    if standardize:
        mean = processed.mean(axis=0, keepdims=True)
        std = processed.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        processed = (processed - mean) / std
    if normalize:
        norms = np.linalg.norm(processed, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        processed = processed / norms
    return processed


def pairwise_distances(points: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        squared_norms = np.sum(points * points, axis=1, keepdims=True)
        squared_distances = squared_norms + squared_norms.T - 2.0 * (points @ points.T)
        np.maximum(squared_distances, 0.0, out=squared_distances)
        return np.sqrt(squared_distances, out=squared_distances)
    if metric == "cosine":
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = points / norms
        similarities = normalized @ normalized.T
        np.clip(similarities, -1.0, 1.0, out=similarities)
        return 1.0 - similarities
    raise ValueError(f"Unsupported metric {metric!r}; expected one of {SUPPORTED_METRICS}")


def whiten_embeddings_for_mahalanobis(
    points: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, dict[str, int | float]]:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0, keepdims=True)
    num_points, original_dim = centered.shape
    if num_points == 0:
        return centered, {
            "original_dim": int(original_dim),
            "covariance_rank": 0,
            "dropped_rank": 0,
        }
    if num_points == 1:
        return np.zeros((1, 1), dtype=np.float64), {
            "original_dim": int(original_dim),
            "covariance_rank": 0,
            "dropped_rank": 0,
        }

    left_singular_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((num_points, 1), dtype=np.float64), {
            "original_dim": int(original_dim),
            "covariance_rank": 0,
            "dropped_rank": 0,
        }

    tolerance = rcond * singular_values[0]
    keep_mask = singular_values > tolerance
    covariance_rank = int(keep_mask.sum())
    if covariance_rank == 0:
        return np.zeros((num_points, 1), dtype=np.float64), {
            "original_dim": int(original_dim),
            "covariance_rank": 0,
            "dropped_rank": int(singular_values.size),
            "largest_singular_value": float(singular_values[0]),
            "smallest_singular_value": float(singular_values[-1]),
        }

    whitened = left_singular_vectors[:, keep_mask] * math.sqrt(num_points - 1)
    retained_singular_values = singular_values[keep_mask]
    return whitened, {
        "original_dim": int(original_dim),
        "covariance_rank": covariance_rank,
        "dropped_rank": int(singular_values.size - covariance_rank),
        "largest_singular_value": float(retained_singular_values[0]),
        "smallest_singular_value": float(retained_singular_values[-1]),
    }


def resolve_geodesic_edge_distance_matrix(
    *,
    points: np.ndarray,
    base_metric_distance_matrix: np.ndarray,
    metric: str,
    geodesic_edge_metric: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if geodesic_edge_metric not in SUPPORTED_GEODESIC_EDGE_METRICS:
        raise ValueError(
            "Unsupported geodesic edge metric "
            f"{geodesic_edge_metric!r}; expected one of {SUPPORTED_GEODESIC_EDGE_METRICS}"
        )

    whitening_summary = None
    resolved_metric = geodesic_edge_metric
    if geodesic_edge_metric == "same_as_metric":
        edge_distance_matrix = np.asarray(base_metric_distance_matrix, dtype=np.float64).copy()
        resolved_metric = metric
    elif geodesic_edge_metric in SUPPORTED_METRICS:
        edge_distance_matrix = pairwise_distances(points, geodesic_edge_metric)
    elif geodesic_edge_metric == "mahalanobis":
        whitened_points, whitening_summary = whiten_embeddings_for_mahalanobis(points)
        edge_distance_matrix = pairwise_distances(whitened_points, "euclidean")
    else:
        raise ValueError(
            "Unsupported geodesic edge metric "
            f"{geodesic_edge_metric!r}; expected one of {SUPPORTED_GEODESIC_EDGE_METRICS}"
        )

    np.fill_diagonal(edge_distance_matrix, 0.0)
    return edge_distance_matrix, {
        "requested_metric": geodesic_edge_metric,
        "resolved_metric": resolved_metric,
        "whitening_summary": whitening_summary,
    }


def summarize_values(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def compute_ordering_probability(
    smaller_values: list[float],
    larger_values: list[float],
) -> dict[str, float | int] | None:
    if not smaller_values or not larger_values:
        return None
    smaller = np.asarray(smaller_values, dtype=np.float64)
    larger = np.asarray(larger_values, dtype=np.float64)
    comparisons = smaller[:, None] < larger[None, :]
    ties = smaller[:, None] == larger[None, :]
    total_pairs = comparisons.size
    return {
        "num_left": int(smaller.size),
        "num_right": int(larger.size),
        "num_pairwise_comparisons": int(total_pairs),
        "probability_left_less_than_right": float(
            (comparisons.sum() + 0.5 * ties.sum()) / total_pairs
        ),
    }


def build_grid_adjacency(
    active_squares: list[int],
    *,
    include_diagonals: bool,
) -> np.ndarray:
    num_active = len(active_squares)
    adjacency = np.zeros((num_active, num_active), dtype=bool)
    for source_idx, source_square in enumerate(active_squares):
        for target_idx in range(source_idx + 1, num_active):
            target_square = active_squares[target_idx]
            if is_local_neighbor(
                source_square,
                target_square,
                include_diagonals=include_diagonals,
            ):
                adjacency[source_idx, target_idx] = True
                adjacency[target_idx, source_idx] = True
    return adjacency


def compute_neighbor_ordering(distance_matrix: np.ndarray) -> np.ndarray:
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(
            "distance_matrix must be square, "
            f"got shape {tuple(distance_matrix.shape)}"
        )
    ordering = np.argsort(distance_matrix, axis=1, kind="stable")
    row_indices = np.arange(distance_matrix.shape[0])[:, None]
    mask = ordering != row_indices
    return ordering[mask].reshape(distance_matrix.shape[0], distance_matrix.shape[0] - 1)


def compute_knn_indices(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k < distance_matrix.shape[0]:
        raise ValueError(
            f"k must lie in [1, {distance_matrix.shape[0] - 1}], got {k}"
        )
    ordering = compute_neighbor_ordering(distance_matrix)
    return ordering[:, :k]


def build_knn_adjacency(
    distance_matrix: np.ndarray,
    *,
    k_by_row: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    knn_adjacency, knn_indices_by_row, _, _ = build_weighted_knn_graph(
        distance_matrix,
        k_by_row=k_by_row,
    )
    return knn_adjacency, knn_indices_by_row


def build_weighted_knn_graph(
    distance_matrix: np.ndarray,
    *,
    k_by_row: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(
            "distance_matrix must be square, "
            f"got shape {tuple(distance_matrix.shape)}"
        )
    if k_by_row.shape != (distance_matrix.shape[0],):
        raise ValueError(
            f"k_by_row must have shape ({distance_matrix.shape[0]},), got {tuple(k_by_row.shape)}"
        )

    ordering = compute_neighbor_ordering(distance_matrix)
    knn_adjacency = np.zeros(distance_matrix.shape, dtype=bool)
    directed_weight_matrix = np.full(distance_matrix.shape, np.inf, dtype=np.float64)
    np.fill_diagonal(directed_weight_matrix, 0.0)
    knn_indices_by_row: list[np.ndarray] = []
    for row_idx, current_k in enumerate(k_by_row.tolist()):
        if not 1 <= int(current_k) < distance_matrix.shape[0]:
            raise ValueError(
                f"Each row k must lie in [1, {distance_matrix.shape[0] - 1}], got {current_k}"
            )
        neighbor_indices = ordering[row_idx, : int(current_k)]
        knn_indices_by_row.append(neighbor_indices)
        knn_adjacency[row_idx, neighbor_indices] = True
        directed_weight_matrix[row_idx, neighbor_indices] = distance_matrix[row_idx, neighbor_indices]
    undirected_weight_matrix = np.minimum(directed_weight_matrix, directed_weight_matrix.T)
    np.fill_diagonal(undirected_weight_matrix, 0.0)
    return knn_adjacency, knn_indices_by_row, directed_weight_matrix, undirected_weight_matrix


def directed_graph_metrics(
    knn_adjacency: np.ndarray,
    directed_grid_adjacency: np.ndarray,
) -> dict[str, float | int]:
    true_positive = int(np.logical_and(knn_adjacency, directed_grid_adjacency).sum())
    predicted_edges = int(knn_adjacency.sum())
    true_edges = int(directed_grid_adjacency.sum())
    precision = true_positive / predicted_edges if predicted_edges > 0 else 0.0
    recall = true_positive / true_edges if true_edges > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    union = int(np.logical_or(knn_adjacency, directed_grid_adjacency).sum())
    return {
        "true_positive_edges": true_positive,
        "predicted_edges": predicted_edges,
        "true_grid_edges": true_edges,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": true_positive / union if union > 0 else 0.0,
    }


def undirected_graph_metrics(
    knn_adjacency: np.ndarray,
    grid_adjacency: np.ndarray,
) -> dict[str, float | int]:
    symmetrized_knn = np.logical_or(knn_adjacency, knn_adjacency.T)
    upper_knn = np.triu(symmetrized_knn, k=1)
    upper_grid = np.triu(grid_adjacency, k=1)
    overlap = np.logical_and(upper_knn, upper_grid)
    true_positive = int(overlap.sum())
    predicted_edges = int(upper_knn.sum())
    true_edges = int(upper_grid.sum())
    precision = true_positive / predicted_edges if predicted_edges > 0 else 0.0
    recall = true_positive / true_edges if true_edges > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    union = int(np.logical_or(upper_knn, upper_grid).sum())
    symmetric_difference = int(np.logical_xor(upper_knn, upper_grid).sum())
    return {
        "true_positive_edges": true_positive,
        "predicted_edges": predicted_edges,
        "true_grid_edges": true_edges,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": true_positive / union if union > 0 else 0.0,
        "edge_symmetric_difference": symmetric_difference,
    }


def all_pairs_shortest_paths(weight_matrix: np.ndarray) -> np.ndarray:
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            "weight_matrix must be square, "
            f"got shape {tuple(weight_matrix.shape)}"
        )
    distances = np.asarray(weight_matrix, dtype=np.float64).copy()
    np.fill_diagonal(distances, 0.0)
    for pivot_idx in range(distances.shape[0]):
        through_pivot = distances[:, [pivot_idx]] + distances[[pivot_idx], :]
        np.minimum(distances, through_pivot, out=distances)
    return distances


def connected_component_sizes_from_weight_matrix(weight_matrix: np.ndarray) -> list[int]:
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError(
            "weight_matrix must be square, "
            f"got shape {tuple(weight_matrix.shape)}"
        )
    adjacency = np.isfinite(weight_matrix)
    np.fill_diagonal(adjacency, False)
    num_nodes = adjacency.shape[0]
    seen = np.zeros(num_nodes, dtype=bool)
    component_sizes = []
    for node_idx in range(num_nodes):
        if seen[node_idx]:
            continue
        stack = [node_idx]
        seen[node_idx] = True
        current_size = 0
        while stack:
            current = stack.pop()
            current_size += 1
            neighbor_indices = np.flatnonzero(adjacency[current])
            for neighbor_idx in neighbor_indices.tolist():
                if seen[neighbor_idx]:
                    continue
                seen[neighbor_idx] = True
                stack.append(neighbor_idx)
        component_sizes.append(current_size)
    component_sizes.sort(reverse=True)
    return component_sizes


def build_unit_weight_matrix(adjacency: np.ndarray) -> np.ndarray:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "adjacency must be square, "
            f"got shape {tuple(adjacency.shape)}"
        )
    weight_matrix = np.full(adjacency.shape, np.inf, dtype=np.float64)
    weight_matrix[adjacency] = 1.0
    np.fill_diagonal(weight_matrix, 0.0)
    return weight_matrix


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start_idx = 0
    while start_idx < array.shape[0]:
        end_idx = start_idx + 1
        while end_idx < array.shape[0] and sorted_values[end_idx] == sorted_values[start_idx]:
            end_idx += 1
        average_rank = 0.5 * (start_idx + end_idx - 1) + 1.0
        ranks[order[start_idx:end_idx]] = average_rank
        start_idx = end_idx
    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "left and right must have the same shape, "
            f"got {left_array.shape} and {right_array.shape}"
        )
    if left_array.size == 0:
        return None
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "left and right must have the same shape, "
            f"got {left_array.shape} and {right_array.shape}"
        )
    if left_array.size == 0:
        return None
    return pearson_correlation(
        average_tie_ranks(left_array),
        average_tie_ranks(right_array),
    )


def compute_stress_metrics(
    *,
    grid_distances: np.ndarray,
    geodesic_distances: np.ndarray,
) -> dict[str, float | None]:
    grid = np.asarray(grid_distances, dtype=np.float64).reshape(-1)
    geodesic = np.asarray(geodesic_distances, dtype=np.float64).reshape(-1)
    if grid.shape != geodesic.shape:
        raise ValueError(
            "grid_distances and geodesic_distances must have the same shape, "
            f"got {grid.shape} and {geodesic.shape}"
        )
    if grid.size == 0:
        return {
            "scale_factor_for_grid": None,
            "normalized_stress": None,
            "rmse": None,
            "mae": None,
        }

    denominator = float(np.dot(grid, grid))
    if denominator > 0.0:
        scale_factor = float(np.dot(geodesic, grid) / denominator)
    else:
        scale_factor = 1.0
    fitted_grid = scale_factor * grid
    residual = geodesic - fitted_grid
    residual_sum_squares = float(np.dot(residual, residual))
    geodesic_sum_squares = float(np.dot(geodesic, geodesic))
    normalized_stress = (
        math.sqrt(residual_sum_squares / geodesic_sum_squares)
        if geodesic_sum_squares > 0.0
        else 0.0
    )
    return {
        "scale_factor_for_grid": scale_factor,
        "normalized_stress": float(normalized_stress),
        "rmse": float(math.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def summarize_rowwise_spearman_correlations(
    *,
    grid_distance_matrix: np.ndarray,
    other_distance_matrix: np.ndarray,
) -> dict[str, object]:
    if grid_distance_matrix.shape != other_distance_matrix.shape:
        raise ValueError(
            "grid_distance_matrix and other_distance_matrix must have the same shape, "
            f"got {grid_distance_matrix.shape} and {other_distance_matrix.shape}"
        )

    rowwise_correlations = []
    num_squares = grid_distance_matrix.shape[0]
    for source_idx in range(num_squares):
        finite_mask = np.isfinite(grid_distance_matrix[source_idx]) & np.isfinite(
            other_distance_matrix[source_idx]
        )
        finite_mask[source_idx] = False
        correlation = spearman_correlation(
            grid_distance_matrix[source_idx, finite_mask],
            other_distance_matrix[source_idx, finite_mask],
        )
        if correlation is not None:
            rowwise_correlations.append(float(correlation))

    summary = summarize_values(rowwise_correlations)
    return {
        "evaluated_square_count": int(len(rowwise_correlations)),
        "total_square_count": int(num_squares),
        "average_spearman_correlation": None if summary is None else float(summary["mean"]),
        "summary": summary,
    }


def build_nearness_rank_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(
            "distance_matrix must be square, "
            f"got shape {tuple(distance_matrix.shape)}"
        )
    num_squares = distance_matrix.shape[0]
    rank_matrix = np.full(distance_matrix.shape, np.nan, dtype=np.float64)
    for source_idx in range(num_squares):
        finite_mask = np.isfinite(distance_matrix[source_idx])
        finite_mask[source_idx] = False
        if not np.any(finite_mask):
            continue
        rank_matrix[source_idx, finite_mask] = average_tie_ranks(
            distance_matrix[source_idx, finite_mask]
        )
    return rank_matrix


def summarize_rowwise_nearness_rank_spearman_correlations(
    *,
    grid_rank_matrix: np.ndarray,
    other_rank_matrix: np.ndarray,
) -> dict[str, object]:
    if grid_rank_matrix.shape != other_rank_matrix.shape:
        raise ValueError(
            "grid_rank_matrix and other_rank_matrix must have the same shape, "
            f"got {grid_rank_matrix.shape} and {other_rank_matrix.shape}"
        )

    rowwise_correlations = []
    num_squares = grid_rank_matrix.shape[0]
    for source_idx in range(num_squares):
        finite_mask = np.isfinite(grid_rank_matrix[source_idx]) & np.isfinite(
            other_rank_matrix[source_idx]
        )
        correlation = pearson_correlation(
            grid_rank_matrix[source_idx, finite_mask],
            other_rank_matrix[source_idx, finite_mask],
        )
        if correlation is not None:
            rowwise_correlations.append(float(correlation))

    summary = summarize_values(rowwise_correlations)
    return {
        "evaluated_square_count": int(len(rowwise_correlations)),
        "total_square_count": int(num_squares),
        "average_spearman_correlation": None if summary is None else float(summary["mean"]),
        "summary": summary,
    }


def compare_nearness_rankings(
    *,
    grid_distance_matrix: np.ndarray,
    other_distance_matrix: np.ndarray,
) -> dict[str, object]:
    if grid_distance_matrix.shape != other_distance_matrix.shape:
        raise ValueError(
            "grid_distance_matrix and other_distance_matrix must have the same shape, "
            f"got {grid_distance_matrix.shape} and {other_distance_matrix.shape}"
        )
    grid_rank_matrix = build_nearness_rank_matrix(grid_distance_matrix)
    other_rank_matrix = build_nearness_rank_matrix(other_distance_matrix)
    breakpoint()
    finite_pair_mask = np.isfinite(grid_rank_matrix) & np.isfinite(other_rank_matrix)
    total_pairs = int(grid_distance_matrix.shape[0] * max(grid_distance_matrix.shape[0] - 1, 0))
    rowwise_spearman = summarize_rowwise_nearness_rank_spearman_correlations(
        grid_rank_matrix=grid_rank_matrix,
        other_rank_matrix=other_rank_matrix,
    )
    return {
        "compared_directed_pair_count": int(finite_pair_mask.sum()),
        "total_directed_pair_count": total_pairs,
        "compared_directed_pair_fraction": (
            float(finite_pair_mask.sum() / total_pairs) if total_pairs > 0 else 0.0
        ),
        "spearman_correlation": pearson_correlation(
            grid_rank_matrix[finite_pair_mask],
            other_rank_matrix[finite_pair_mask],
        ),
        "rowwise_spearman_correlation": rowwise_spearman,
    }


def compare_distance_matrices(
    *,
    grid_distance_matrix: np.ndarray,
    other_distance_matrix: np.ndarray,
) -> dict[str, object]:
    if grid_distance_matrix.shape != other_distance_matrix.shape:
        raise ValueError(
            "grid_distance_matrix and other_distance_matrix must have the same shape, "
            f"got {grid_distance_matrix.shape} and {other_distance_matrix.shape}"
        )
    upper_mask = np.triu(np.ones(grid_distance_matrix.shape, dtype=bool), k=1)
    finite_pair_mask = upper_mask & np.isfinite(grid_distance_matrix) & np.isfinite(other_distance_matrix)
    grid_values = grid_distance_matrix[finite_pair_mask]
    other_values = other_distance_matrix[finite_pair_mask]
    total_pairs = int(upper_mask.sum())
    stress_metrics = compute_stress_metrics(
        grid_distances=grid_values,
        geodesic_distances=other_values,
    )
    rowwise_spearman = summarize_rowwise_spearman_correlations(
        grid_distance_matrix=grid_distance_matrix,
        other_distance_matrix=other_distance_matrix,
    )
    return {
        "compared_pair_count": int(grid_values.size),
        "total_pair_count": total_pairs,
        "compared_pair_fraction": float(grid_values.size / total_pairs) if total_pairs > 0 else 0.0,
        "spearman_correlation": spearman_correlation(grid_values, other_values),
        "rowwise_spearman_correlation": rowwise_spearman,
        "distance_summary": summarize_values(other_values.tolist()),
        "stress": stress_metrics,
    }


def geodesic_neighbor_retrieval_metrics(
    *,
    geodesic_distance_matrix: np.ndarray,
    grid_adjacency: np.ndarray,
) -> dict[str, float | int | None]:
    if geodesic_distance_matrix.shape != grid_adjacency.shape:
        raise ValueError(
            "geodesic_distance_matrix and grid_adjacency must have the same shape, "
            f"got {geodesic_distance_matrix.shape} and {grid_adjacency.shape}"
        )
    grid_degrees = grid_adjacency.sum(axis=1).astype(int)
    top1_hits = []
    hit_counts = []
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for source_idx in range(geodesic_distance_matrix.shape[0]):
        reachable_mask = np.isfinite(geodesic_distance_matrix[source_idx])
        reachable_mask[source_idx] = False
        if not np.any(reachable_mask):
            continue
        ordering = np.argsort(geodesic_distance_matrix[source_idx], kind="stable")
        neighbor_indices = [
            int(target_idx)
            for target_idx in ordering.tolist()
            if target_idx != source_idx and reachable_mask[target_idx]
        ]
        if not neighbor_indices:
            continue
        top1_hits.append(bool(grid_adjacency[source_idx, neighbor_indices[0]]))

        grid_degree = int(grid_degrees[source_idx])
        if grid_degree > 0:
            top_k = min(grid_degree, len(neighbor_indices))
            selected_neighbor_indices = neighbor_indices[:top_k]
            hit_count = int(grid_adjacency[source_idx, selected_neighbor_indices].sum())
            hit_counts.append(hit_count)
            precision_scores.append(hit_count / top_k if top_k > 0 else 0.0)
            recall_scores.append(hit_count / grid_degree)
            true_neighbor_ranks = [
                rank_idx + 1
                for rank_idx, target_idx in enumerate(neighbor_indices)
                if grid_adjacency[source_idx, target_idx]
            ]
            if true_neighbor_ranks:
                reciprocal_ranks.append(1.0 / min(true_neighbor_ranks))

    evaluated_square_count = len(top1_hits)
    return {
        "evaluated_square_count": int(evaluated_square_count),
        "top1_is_true_local_neighbor_rate": (
            float(np.mean(top1_hits)) if top1_hits else None
        ),
        "mean_reciprocal_rank_first_true_local_neighbor": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        ),
        "mean_grid_neighbor_hits_at_true_degree": (
            float(np.mean(hit_counts)) if hit_counts else None
        ),
        "mean_grid_neighbor_precision_at_true_degree": (
            float(np.mean(precision_scores)) if precision_scores else None
        ),
        "mean_grid_neighbor_recall_at_true_degree": (
            float(np.mean(recall_scores)) if recall_scores else None
        ),
    }


def evaluate_geodesic_k_value(
    *,
    k: int | None,
    active_squares: list[int],
    edge_distance_matrix: np.ndarray,
    grid_adjacency: np.ndarray,
    grid_shortest_path_distances: np.ndarray,
    match_groundtruth_degree: bool,
    geodesic_metric_info: dict[str, object],
) -> dict[str, object]:
    num_active = len(active_squares)
    grid_degrees = grid_adjacency.sum(axis=1).astype(int)
    if match_groundtruth_degree:
        k_by_row = grid_degrees.copy()
    else:
        if k is None:
            raise ValueError("Fixed-k geodesic evaluation requires an integer k")
        k_by_row = np.full(num_active, int(k), dtype=int)

    _, _, _, undirected_weight_matrix = build_weighted_knn_graph(
        edge_distance_matrix,
        k_by_row=k_by_row,
    )
    geodesic_distance_matrix = all_pairs_shortest_paths(undirected_weight_matrix)
    component_sizes = connected_component_sizes_from_weight_matrix(undirected_weight_matrix)
    geodesic_comparison = compare_distance_matrices(
        grid_distance_matrix=grid_shortest_path_distances,
        other_distance_matrix=geodesic_distance_matrix,
    )
    retrieval_metrics = geodesic_neighbor_retrieval_metrics(
        geodesic_distance_matrix=geodesic_distance_matrix,
        grid_adjacency=grid_adjacency,
    )

    return {
        "k": None if k is None else int(k),
        "k_mode": "groundtruth_degree" if match_groundtruth_degree else "fixed",
        "effective_k_summary": {
            "min": int(k_by_row.min()),
            "max": int(k_by_row.max()),
            "mean": float(k_by_row.mean()),
        },
        "effective_k_histogram": {
            str(int(degree)): int((k_by_row == degree).sum())
            for degree in sorted(set(k_by_row.tolist()))
        },
        "edge_metric": geodesic_metric_info["resolved_metric"],
        "requested_edge_metric": geodesic_metric_info["requested_metric"],
        "edge_metric_whitening_summary": geodesic_metric_info["whitening_summary"],
        "graph_symmetrization": "union_of_directed_knn_edges",
        "component_count": int(len(component_sizes)),
        "largest_component_size": int(component_sizes[0]) if component_sizes else 0,
        "component_sizes": [int(size) for size in component_sizes],
        "geodesic_grid_comparison": geodesic_comparison,
        "nearest_neighbor_retrieval": retrieval_metrics,
    }


def evaluate_k_value(
    *,
    k: int | None,
    active_squares: list[int],
    distance_matrix: np.ndarray,
    grid_adjacency: np.ndarray,
    include_diagonals: bool,
    match_groundtruth_degree: bool,
    include_per_square: bool,
) -> dict[str, object]:
    num_active = len(active_squares)
    directed_grid_adjacency = grid_adjacency.copy()
    grid_degrees = grid_adjacency.sum(axis=1).astype(int)
    if match_groundtruth_degree:
        k_by_row = grid_degrees.copy()
    else:
        if k is None:
            raise ValueError("Fixed-k evaluation requires an integer k")
        k_by_row = np.full(num_active, int(k), dtype=int)
    knn_adjacency, knn_indices_by_row = build_knn_adjacency(
        distance_matrix,
        k_by_row=k_by_row,
    )

    per_square_results = []
    hit_counts = []
    precision_scores = []
    recall_scores = []
    same_row_far_counts = []
    same_col_far_counts = []
    same_diagonal_far_counts = []
    unrelated_counts = []

    for source_idx, source_square in enumerate(active_squares):
        neighbor_indices = knn_indices_by_row[source_idx]
        neighbor_squares = [active_squares[int(idx)] for idx in neighbor_indices]
        current_k = int(k_by_row[source_idx])
        grid_neighbor_count = 0
        same_row_far_count = 0
        same_col_far_count = 0
        same_diagonal_far_count = 0
        unrelated_count = 0
        neighbor_records = []

        for target_idx, target_square in zip(neighbor_indices, neighbor_squares):
            category = classify_square_pair(
                source_square,
                target_square,
                include_diagonals=include_diagonals,
            )
            is_grid_neighbor = category == "grid_neighbor"
            grid_neighbor_count += int(is_grid_neighbor)
            same_row_far_count += int(category == "same_row_far")
            same_col_far_count += int(category == "same_col_far")
            same_diagonal_far_count += int(category == "same_diagonal_far")
            unrelated_count += int(category == "unrelated")
            if include_per_square:
                neighbor_records.append(
                    {
                        "square_index": int(target_square),
                        "square_label": square_label(int(target_square)),
                        "distance": float(distance_matrix[source_idx, int(target_idx)]),
                        "category": category,
                        "is_true_local_neighbor": bool(is_grid_neighbor),
                        "board_manhattan_distance": int(
                            board_manhattan_distance(source_square, target_square)
                        ),
                        "board_chebyshev_distance": int(
                            board_chebyshev_distance(source_square, target_square)
                        ),
                    }
                )

        grid_degree = int(grid_degrees[source_idx])
        precision = grid_neighbor_count / current_k if current_k > 0 else 0.0
        recall = grid_neighbor_count / grid_degree if grid_degree > 0 else 0.0
        hit_counts.append(grid_neighbor_count)
        precision_scores.append(precision)
        recall_scores.append(recall)
        same_row_far_counts.append(same_row_far_count)
        same_col_far_counts.append(same_col_far_count)
        same_diagonal_far_counts.append(same_diagonal_far_count)
        unrelated_counts.append(unrelated_count)

        if include_per_square:
            true_local_neighbor_squares = [
                active_squares[target_idx]
                for target_idx, is_edge in enumerate(grid_adjacency[source_idx])
                if is_edge
            ]
            per_square_results.append(
                {
                    "square_index": int(source_square),
                    "square_label": square_label(int(source_square)),
                    "effective_k": current_k,
                    "grid_degree": grid_degree,
                    "grid_neighbor_hits": int(grid_neighbor_count),
                    "grid_neighbor_precision_at_k": float(precision),
                    "grid_neighbor_recall": float(recall),
                    "same_row_far_count": int(same_row_far_count),
                    "same_col_far_count": int(same_col_far_count),
                    "same_diagonal_far_count": int(same_diagonal_far_count),
                    "unrelated_count": int(unrelated_count),
                    "true_local_neighbors": [
                        {
                            "square_index": int(target_square),
                            "square_label": square_label(int(target_square)),
                        }
                        for target_square in true_local_neighbor_squares
                    ],
                    "embedding_knn_neighbors": neighbor_records,
                    "neighbors": neighbor_records,
                }
            )

    k_mode = "groundtruth_degree" if match_groundtruth_degree else "fixed"
    result = {
        "k": None if k is None else int(k),
        "k_mode": k_mode,
        "effective_k_summary": {
            "min": int(k_by_row.min()),
            "max": int(k_by_row.max()),
            "mean": float(k_by_row.mean()),
        },
        "effective_k_histogram": {
            str(int(degree)): int((k_by_row == degree).sum())
            for degree in sorted(set(k_by_row.tolist()))
        },
        "local_summary": {
            "mean_grid_neighbor_hits": float(np.mean(hit_counts)),
            "mean_grid_neighbor_precision_at_k": float(np.mean(precision_scores)),
            "mean_grid_neighbor_recall": float(np.mean(recall_scores)),
            "mean_same_row_far_count": float(np.mean(same_row_far_counts)),
            "mean_same_col_far_count": float(np.mean(same_col_far_counts)),
            "mean_same_diagonal_far_count": float(np.mean(same_diagonal_far_counts)),
            "mean_unrelated_count": float(np.mean(unrelated_counts)),
        },
        "directed_graph": directed_graph_metrics(knn_adjacency, directed_grid_adjacency),
        "undirected_graph": undirected_graph_metrics(knn_adjacency, grid_adjacency),
    }
    if include_per_square:
        result["per_square"] = per_square_results
    return result


def category_distance_analysis(
    *,
    active_squares: list[int],
    distance_matrix: np.ndarray,
    include_diagonals: bool,
) -> dict[str, object]:
    distances_by_category = {
        "grid_neighbor": [],
        "same_row_far": [],
        "same_col_far": [],
        "same_diagonal_far": [],
        "same_axis_far": [],
        "unrelated": [],
    }

    for left_idx, left_square in enumerate(active_squares):
        for right_idx in range(left_idx + 1, len(active_squares)):
            right_square = active_squares[right_idx]
            category = classify_square_pair(
                left_square,
                right_square,
                include_diagonals=include_diagonals,
            )
            distance = float(distance_matrix[left_idx, right_idx])
            distances_by_category[category].append(distance)
            if category in {"same_row_far", "same_col_far"}:
                distances_by_category["same_axis_far"].append(distance)

    summaries = {
        category: summarize_values(values)
        for category, values in distances_by_category.items()
    }
    ordering_probabilities = {
        "grid_neighbor_vs_same_axis_far": compute_ordering_probability(
            distances_by_category["grid_neighbor"],
            distances_by_category["same_axis_far"],
        ),
        "grid_neighbor_vs_same_row_far": compute_ordering_probability(
            distances_by_category["grid_neighbor"],
            distances_by_category["same_row_far"],
        ),
        "grid_neighbor_vs_same_col_far": compute_ordering_probability(
            distances_by_category["grid_neighbor"],
            distances_by_category["same_col_far"],
        ),
        "grid_neighbor_vs_same_diagonal_far": compute_ordering_probability(
            distances_by_category["grid_neighbor"],
            distances_by_category["same_diagonal_far"],
        ),
        "grid_neighbor_vs_unrelated": compute_ordering_probability(
            distances_by_category["grid_neighbor"],
            distances_by_category["unrelated"],
        ),
        "same_axis_far_vs_unrelated": compute_ordering_probability(
            distances_by_category["same_axis_far"],
            distances_by_category["unrelated"],
        ),
        "same_row_far_vs_unrelated": compute_ordering_probability(
            distances_by_category["same_row_far"],
            distances_by_category["unrelated"],
        ),
        "same_col_far_vs_unrelated": compute_ordering_probability(
            distances_by_category["same_col_far"],
            distances_by_category["unrelated"],
        ),
        "same_diagonal_far_vs_unrelated": compute_ordering_probability(
            distances_by_category["same_diagonal_far"],
            distances_by_category["unrelated"],
        ),
    }
    return {
        "distance_summaries": summaries,
        "ordering_probabilities": ordering_probabilities,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-path",
        required=True,
        help="Path to a saved bilinear or trilinear TPR probe.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional JSON output path. Defaults next to the probe checkpoint.",
    )
    parser.add_argument(
        "--square-embedding-source",
        choices=SUPPORTED_SQUARE_EMBEDDING_SOURCES,
        default=RoleGridGraphConfig.square_embedding_source,
        help=(
            "How to derive one square embedding per board cell. `auto` uses direct "
            "`role_embeddings` when present and otherwise builds each square embedding "
            "from the flattened outer product of `row_embeddings` and `col_embeddings`."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default=RoleGridGraphConfig.metric,
        help="Distance metric used for k-NN and category comparisons.",
    )
    parser.add_argument(
        "--geodesic-edge-metric",
        choices=SUPPORTED_GEODESIC_EDGE_METRICS,
        default=RoleGridGraphConfig.geodesic_edge_metric,
        help=(
            "Local edge metric for the Isomap-style geodesic analysis. "
            "`mahalanobis` whitens the square embeddings first; `same_as_metric` "
            "reuses the main --metric distance."
        ),
    )
    parser.add_argument(
        "--mean-center",
        action="store_true",
        help=(
            "Subtract the mean square embedding vector before computing distances. "
            "This is especially useful with --metric cosine because it removes the "
            "global offset before the cosine comparison."
        ),
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
        help="Exclude the four starting center squares from the graph test.",
    )
    parser.add_argument(
        "--include-diagonals",
        action="store_true",
        help=(
            "Treat diagonal-adjacent squares as valid local neighbors too, "
            "so the reference board graph uses the 8-neighborhood instead of the 4-neighborhood."
        ),
    )
    parser.add_argument(
        "--k",
        dest="k_values",
        type=int,
        action="append",
        default=None,
        help="k for the square-embedding k-NN graph. May be passed multiple times.",
    )
    parser.add_argument(
        "--match-groundtruth-degree",
        action="store_true",
        help=(
            "Set k dynamically per square to that square's true board-neighbor count "
            "after filtering. This automatically uses degree 2/3/4 in 4-neighborhood "
            "mode and degree 3/5/8 in 8-neighborhood mode, with further changes if "
            "squares are excluded."
        ),
    )
    parser.add_argument(
        "--include-per-square",
        action="store_true",
        help="Include per-square neighbor breakdowns in the JSON output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.match_groundtruth_degree and args.k_values is not None:
        raise ValueError("--match-groundtruth-degree cannot be combined with explicit --k")
    config = RoleGridGraphConfig(
        probe_path=args.probe_path,
        output_path=args.output_path,
        square_embedding_source=args.square_embedding_source,
        metric=args.metric,
        geodesic_edge_metric=args.geodesic_edge_metric,
        mean_center=args.mean_center,
        standardize=args.standardize,
        normalize=args.normalize,
        exclude_center_squares=args.exclude_center_squares,
        include_diagonals=args.include_diagonals,
        k_values=tuple(() if args.match_groundtruth_degree else (args.k_values or DEFAULT_K_VALUES)),
        match_groundtruth_degree=args.match_groundtruth_degree,
        include_per_square=args.include_per_square,
    )

    probe_path = Path(config.probe_path).expanduser().resolve()
    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_path}")
    output_path = resolve_output_path(config, probe_path)

    square_embeddings, artifact = load_square_embeddings(
        probe_path,
        square_embedding_source=config.square_embedding_source,
    )
    active_squares = [
        square_idx
        for square_idx in range(NUM_SQUARES)
        if not (config.exclude_center_squares and square_idx in STARTING_SQUARES)
    ]
    if len(active_squares) < 2:
        raise ValueError("Need at least two active squares to build a k-NN graph")

    max_valid_k = len(active_squares) - 1
    k_values = sorted({int(k) for k in config.k_values})
    processed_embeddings = preprocess_embeddings(
        square_embeddings[active_squares],
        mean_center=config.mean_center,
        standardize=config.standardize,
        normalize=config.normalize,
    )
    distance_matrix = pairwise_distances(processed_embeddings, config.metric)
    geodesic_edge_distance_matrix, geodesic_metric_info = resolve_geodesic_edge_distance_matrix(
        points=processed_embeddings,
        base_metric_distance_matrix=distance_matrix,
        metric=config.metric,
        geodesic_edge_metric=config.geodesic_edge_metric,
    )
    grid_adjacency = build_grid_adjacency(
        active_squares,
        include_diagonals=config.include_diagonals,
    )
    grid_shortest_path_distances = all_pairs_shortest_paths(
        build_unit_weight_matrix(grid_adjacency)
    )
    if config.match_groundtruth_degree:
        min_grid_degree = int(grid_adjacency.sum(axis=1).min())
        max_grid_degree = int(grid_adjacency.sum(axis=1).max())
        if min_grid_degree <= 0:
            raise ValueError(
                "Ground-truth degree matching requires every active square to have at least one neighbor"
            )
    else:
        invalid_k_values = [k for k in k_values if not 1 <= k <= max_valid_k]
        if invalid_k_values:
            raise ValueError(
                f"k must lie in [1, {max_valid_k}] after square filtering; got {invalid_k_values}"
            )

    category_analysis = category_distance_analysis(
        active_squares=active_squares,
        distance_matrix=distance_matrix,
        include_diagonals=config.include_diagonals,
    )
    embedding_grid_distance_comparison = compare_distance_matrices(
        grid_distance_matrix=grid_shortest_path_distances,
        other_distance_matrix=distance_matrix,
    )
    embedding_grid_distance_comparison["nearness_ranking_comparison"] = compare_nearness_rankings(
        grid_distance_matrix=grid_shortest_path_distances,
        other_distance_matrix=distance_matrix,
    )
    if config.match_groundtruth_degree:
        knn_results = [
            evaluate_k_value(
                k=None,
                active_squares=active_squares,
                distance_matrix=distance_matrix,
                grid_adjacency=grid_adjacency,
                include_diagonals=config.include_diagonals,
                match_groundtruth_degree=True,
                include_per_square=config.include_per_square,
            )
        ]
        geodesic_results = [
            evaluate_geodesic_k_value(
                k=None,
                active_squares=active_squares,
                edge_distance_matrix=geodesic_edge_distance_matrix,
                grid_adjacency=grid_adjacency,
                grid_shortest_path_distances=grid_shortest_path_distances,
                match_groundtruth_degree=True,
                geodesic_metric_info=geodesic_metric_info,
            )
        ]
    else:
        knn_results = [
            evaluate_k_value(
                k=k,
                active_squares=active_squares,
                distance_matrix=distance_matrix,
                grid_adjacency=grid_adjacency,
                include_diagonals=config.include_diagonals,
                match_groundtruth_degree=False,
                include_per_square=config.include_per_square,
            )
            for k in k_values
        ]
        geodesic_results = [
            evaluate_geodesic_k_value(
                k=k,
                active_squares=active_squares,
                edge_distance_matrix=geodesic_edge_distance_matrix,
                grid_adjacency=grid_adjacency,
                grid_shortest_path_distances=grid_shortest_path_distances,
                match_groundtruth_degree=False,
                geodesic_metric_info=geodesic_metric_info,
            )
            for k in k_values
        ]

    result = {
        "probe_path": str(probe_path),
        "config": asdict(config),
        "probe_metadata": build_probe_metadata(artifact),
        "active_square_count": len(active_squares),
        "active_squares": [
            {
                "square_index": int(square_idx),
                "square_label": square_label(square_idx),
            }
            for square_idx in active_squares
        ],
        "grid_edge_count": int(np.triu(grid_adjacency, k=1).sum()),
        "category_analysis": category_analysis,
        "embedding_grid_distance_comparison": embedding_grid_distance_comparison,
        "knn_results": knn_results,
        "geodesic_results": geodesic_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("Role grid-graph test config:")
    print(json.dumps(asdict(config), indent=2))
    print("Probe metadata:")
    print(json.dumps(result["probe_metadata"], indent=2))
    print(
        f"Active squares: {len(active_squares)} | "
        f"grid edges: {result['grid_edge_count']} | "
        f"metric: {config.metric} | "
        f"geodesic_edge_metric: {geodesic_metric_info['resolved_metric']} | "
        f"mean_center: {config.mean_center} | "
        f"local-neighbor mode: {'8-neighborhood' if config.include_diagonals else '4-neighborhood'}"
    )
    print("Category distance summary:")
    for category_name in (
        "grid_neighbor",
        "same_row_far",
        "same_col_far",
        "same_diagonal_far",
        "same_axis_far",
        "unrelated",
    ):
        summary = category_analysis["distance_summaries"][category_name]
        if summary is None:
            print(f"  {category_name}: no pairs")
            continue
        print(
            f"  {category_name}: "
            f"mean={summary['mean']:.4f} "
            f"median={summary['median']:.4f} "
            f"q25={summary['q25']:.4f} "
            f"q75={summary['q75']:.4f} "
            f"count={summary['count']}"
        )
    print("Ordering probabilities:")
    for comparison_name, metrics in category_analysis["ordering_probabilities"].items():
        if metrics is None:
            print(f"  {comparison_name}: unavailable")
            continue
        print(
            f"  {comparison_name}: "
            f"{metrics['probability_left_less_than_right']:.4f}"
        )
    embedding_spearman = embedding_grid_distance_comparison["spearman_correlation"]
    embedding_rowwise_summary = embedding_grid_distance_comparison["rowwise_spearman_correlation"]
    embedding_rowwise_spearman = embedding_rowwise_summary["average_spearman_correlation"]
    embedding_nearness_ranking = embedding_grid_distance_comparison["nearness_ranking_comparison"]
    embedding_nearness_rank_spearman = embedding_nearness_ranking["spearman_correlation"]
    print(
        "Embedding-vs-grid distance comparison: "
        f"pair_spearman={'n/a' if embedding_spearman is None else f'{embedding_spearman:.3f}'} "
        "nearness_rank_spearman="
        f"{'n/a' if embedding_nearness_rank_spearman is None else f'{embedding_nearness_rank_spearman:.3f}'} "
        "mean_per_square_spearman="
        f"{'n/a' if embedding_rowwise_spearman is None else f'{embedding_rowwise_spearman:.3f}'} "
        "evaluated_squares="
        f"{embedding_rowwise_summary['evaluated_square_count']}/"
        f"{embedding_rowwise_summary['total_square_count']}"
    )
    print("k-NN graph results:")
    for knn_result in knn_results:
        local = knn_result["local_summary"]
        directed = knn_result["directed_graph"]
        undirected = knn_result["undirected_graph"]
        if knn_result["k_mode"] == "groundtruth_degree":
            k_label = (
                "k=groundtruth_degree"
                f"[mean={knn_result['effective_k_summary']['mean']:.2f},"
                f"min={knn_result['effective_k_summary']['min']},"
                f"max={knn_result['effective_k_summary']['max']}]"
            )
        else:
            k_label = f"k={knn_result['k']}"
        print(
            f"  {k_label}: "
            f"local_hits={local['mean_grid_neighbor_hits']:.3f} "
            f"local_precision={local['mean_grid_neighbor_precision_at_k']:.3f} "
            f"local_recall={local['mean_grid_neighbor_recall']:.3f} "
            f"same_diag_far={local['mean_same_diagonal_far_count']:.3f} "
            f"undirected_precision={undirected['precision']:.3f} "
            f"undirected_recall={undirected['recall']:.3f} "
            f"undirected_jaccard={undirected['jaccard']:.3f} "
            f"directed_precision={directed['precision']:.3f} "
            f"directed_recall={directed['recall']:.3f}"
        )
    print("Isomap-style geodesic results:")
    for geodesic_result in geodesic_results:
        comparison = geodesic_result["geodesic_grid_comparison"]
        retrieval = geodesic_result["nearest_neighbor_retrieval"]
        stress = comparison["stress"]
        if geodesic_result["k_mode"] == "groundtruth_degree":
            k_label = (
                "k=groundtruth_degree"
                f"[mean={geodesic_result['effective_k_summary']['mean']:.2f},"
                f"min={geodesic_result['effective_k_summary']['min']},"
                f"max={geodesic_result['effective_k_summary']['max']}]"
            )
        else:
            k_label = f"k={geodesic_result['k']}"
        spearman_text = (
            "n/a"
            if comparison["spearman_correlation"] is None
            else f"{comparison['spearman_correlation']:.3f}"
        )
        stress_text = (
            "n/a"
            if stress["normalized_stress"] is None
            else f"{stress['normalized_stress']:.3f}"
        )
        top1_text = (
            "n/a"
            if retrieval["top1_is_true_local_neighbor_rate"] is None
            else f"{retrieval['top1_is_true_local_neighbor_rate']:.3f}"
        )
        recall_text = (
            "n/a"
            if retrieval["mean_grid_neighbor_recall_at_true_degree"] is None
            else f"{retrieval['mean_grid_neighbor_recall_at_true_degree']:.3f}"
        )
        print(
            f"  {k_label}: "
            f"components={geodesic_result['component_count']} "
            f"largest_component={geodesic_result['largest_component_size']} "
            f"pair_coverage={comparison['compared_pair_fraction']:.3f} "
            f"spearman={spearman_text} "
            f"stress={stress_text} "
            f"top1_neighbor={top1_text} "
            f"true_degree_recall={recall_text}"
        )
    print(f"Wrote role grid-graph analysis to {output_path}")


if __name__ == "__main__":
    main()
