"""Create interactive PCA plots for TPR role and filler embeddings."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

import numpy as np
import torch


STATE_LABELS = ("empty", "opponent", "current")
ROW_LABELS = "ABCDEFGH"
BOARD_ROWS = 8
BOARD_COLS = 8
STARTING_SQUARES = frozenset({27, 28, 35, 36})
ROW_BASE_COLORS = (
    "#E41A1C",
    "#FF7F00",
    "#FFD92F",
    "#4DAF4A",
    "#00BFC4",
    "#377EB8",
    "#4B0082",
    "#984EA3",
)


@dataclass
class ProbeEmbeddingRecord:
    probe_path: Path
    layer: int
    role_dim: int
    filler_dim: int
    use_bias: bool
    role_embeddings: np.ndarray
    filler_embeddings: np.ndarray


def square_label(index: int) -> str:
    row_idx, col_idx = square_row_col(index)
    return f"{ROW_LABELS[row_idx]}{col_idx + 1}"


def square_row_col(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_COLS)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_value = hex_color.lstrip("#")
    if len(hex_value) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return tuple(int(hex_value[idx : idx + 2], 16) for idx in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        *(max(0, min(255, round(channel))) for channel in rgb)
    )


def blend_rgb(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    mix: float,
) -> tuple[float, float, float]:
    return tuple(
        (1.0 - mix) * start_channel + mix * end_channel
        for start_channel, end_channel in zip(start_rgb, end_rgb)
    )


def row_column_color(row_idx: int, col_idx: int) -> str:
    base_rgb = hex_to_rgb(ROW_BASE_COLORS[row_idx])
    light_rgb = blend_rgb(base_rgb, (255, 255, 255), 0.55)
    dark_rgb = blend_rgb(base_rgb, (0, 0, 0), 0.2)
    mix = col_idx / (BOARD_COLS - 1) if BOARD_COLS > 1 else 0.0
    return rgb_to_hex(blend_rgb(light_rgb, dark_rgb, mix))


def row_legend_color(row_idx: int) -> str:
    base_rgb = hex_to_rgb(ROW_BASE_COLORS[row_idx])
    return rgb_to_hex(blend_rgb(base_rgb, (0, 0, 0), 0.2))


def board_neighbor_pairs(
    excluded_square_indices: Collection[int] | None = None,
) -> list[tuple[int, int]]:
    excluded = set(excluded_square_indices or ())
    pairs: list[tuple[int, int]] = []
    for row_idx in range(BOARD_ROWS):
        for col_idx in range(BOARD_COLS):
            source_square = row_idx * BOARD_COLS + col_idx
            if source_square in excluded:
                continue
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    if row_delta == 0 and col_delta == 0:
                        continue
                    neighbor_row = row_idx + row_delta
                    neighbor_col = col_idx + col_delta
                    if not (0 <= neighbor_row < BOARD_ROWS):
                        continue
                    if not (0 <= neighbor_col < BOARD_COLS):
                        continue
                    target_square = neighbor_row * BOARD_COLS + neighbor_col
                    if target_square in excluded or target_square <= source_square:
                        continue
                    pairs.append((source_square, target_square))
    return pairs


def build_board_neighbor_line_coordinates(
    *,
    points_by_record_and_square: Mapping[int, Mapping[int, np.ndarray]],
    n_plot_components: int,
    excluded_square_indices: Collection[int] | None = None,
) -> tuple[list[float | None], list[float | None], list[float | None] | None]:
    x_coords: list[float | None] = []
    y_coords: list[float | None] = []
    z_coords: list[float | None] | None = [] if n_plot_components == 3 else None
    for square_points in points_by_record_and_square.values():
        for source_square, target_square in board_neighbor_pairs(
            excluded_square_indices=excluded_square_indices
        ):
            source_point = square_points.get(source_square)
            target_point = square_points.get(target_square)
            if source_point is None or target_point is None:
                continue
            x_coords.extend([float(source_point[0]), float(target_point[0]), None])
            y_coords.extend([float(source_point[1]), float(target_point[1]), None])
            if z_coords is not None:
                z_coords.extend([float(source_point[2]), float(target_point[2]), None])
    return x_coords, y_coords, z_coords


def add_board_neighbor_line_trace(
    figure,
    go,
    *,
    subplot_row: int,
    subplot_col: int,
    points_by_record_and_square: Mapping[int, Mapping[int, np.ndarray]],
    n_plot_components: int,
    excluded_square_indices: Collection[int] | None = None,
) -> None:
    x_coords, y_coords, z_coords = build_board_neighbor_line_coordinates(
        points_by_record_and_square=points_by_record_and_square,
        n_plot_components=n_plot_components,
        excluded_square_indices=excluded_square_indices,
    )
    if not x_coords:
        return

    trace_kwargs = {
        "x": x_coords,
        "y": y_coords,
        "mode": "lines",
        "line": {
            "color": "#000000",
            "width": 0.7 if n_plot_components == 3 else 1.25,
        },
        "hoverinfo": "skip",
        "connectgaps": False,
        "showlegend": False,
        "name": "Board neighbors",
    }
    if n_plot_components == 3:
        figure.add_trace(
            go.Scatter3d(z=z_coords, **trace_kwargs),
            row=subplot_row,
            col=subplot_col,
        )
    else:
        figure.add_trace(
            go.Scatter(**trace_kwargs),
            row=subplot_row,
            col=subplot_col,
        )


def parse_int_list(raw_value: str | None) -> set[int] | None:
    if raw_value is None:
        return None
    values = {int(piece.strip()) for piece in raw_value.split(",") if piece.strip()}
    if not values:
        raise ValueError("Expected at least one integer in the comma-separated list")
    return values


def parse_principal_components(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None

    components = tuple(
        int(piece.strip()) for piece in raw_value.split(",") if piece.strip()
    )
    if len(components) not in (2, 3):
        raise ValueError(
            "Expected exactly 2 or 3 principal components in the comma-separated list"
        )
    if any(component <= 0 for component in components):
        raise ValueError("Principal components must be positive integers")
    if len(set(components)) != len(components):
        raise ValueError("Principal components must be distinct")
    return components


def discover_probe_paths(
    input_paths: Iterable[str],
    layers: set[int] | None,
    role_dims: set[int] | None,
    filler_dims: set[int] | None,
    include_bias: bool,
) -> list[Path]:
    paths = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("resid_*_tpr_*.pth"))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

        for candidate in candidates:
            artifact = torch.load(candidate, map_location="cpu")
            layer = int(artifact["layer"])
            role_dim = int(artifact["role_dim"])
            filler_dim = int(artifact["filler_dim"])
            use_bias = bool(artifact.get("use_bias", False))
            if layers is not None and layer not in layers:
                continue
            if role_dims is not None and role_dim not in role_dims:
                continue
            if filler_dims is not None and filler_dim not in filler_dims:
                continue
            if not include_bias and use_bias:
                continue
            paths.append(candidate)

    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise ValueError("No matching TPR probe checkpoints found")
    return unique_paths


def load_probe_embedding_records(probe_paths: Iterable[Path]) -> list[ProbeEmbeddingRecord]:
    records = []
    for probe_path in probe_paths:
        artifact = torch.load(probe_path, map_location="cpu")
        state_dict = artifact["probe_state_dict"]
        role_embeddings = (
            state_dict["role_embeddings"].detach().cpu().to(torch.float32).numpy()
        )
        filler_embeddings = (
            state_dict["filler_embeddings"].detach().cpu().to(torch.float32).numpy()
        )
        records.append(
            ProbeEmbeddingRecord(
                probe_path=probe_path,
                layer=int(artifact["layer"]),
                role_dim=int(artifact["role_dim"]),
                filler_dim=int(artifact["filler_dim"]),
                use_bias=bool(artifact.get("use_bias", False)),
                role_embeddings=role_embeddings.reshape(BOARD_ROWS * BOARD_COLS, -1),
                filler_embeddings=filler_embeddings.reshape(len(STATE_LABELS), -1),
            )
        )
    return records


def maybe_standardize_points(
    points: np.ndarray,
    *,
    standardize: bool,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if not standardize:
        return points
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for --standardize. Install scikit-learn first."
        ) from exc
    standardized = StandardScaler().fit_transform(points)
    return np.asarray(standardized, dtype=np.float32)


def compute_pca(
    points: np.ndarray,
    n_components: int,
    *,
    standardize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    points = maybe_standardize_points(points, standardize=standardize)
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


def pca_components_for_embedding_dim(
    embedding_dim: int,
    force_2d: bool = False,
) -> int:
    if force_2d:
        return 2
    return 3 if embedding_dim >= 3 else 2


def default_principal_components_for_embedding_dim(
    embedding_dim: int,
    force_2d: bool = False,
) -> tuple[int, ...]:
    n_components = pca_components_for_embedding_dim(
        embedding_dim=embedding_dim,
        force_2d=force_2d,
    )
    return tuple(range(1, n_components + 1))


def resolve_principal_components_for_embedding_dim(
    embedding_dim: int,
    *,
    principal_components: tuple[int, ...] | None,
    force_2d: bool = False,
) -> tuple[int, ...]:
    if principal_components is None:
        return default_principal_components_for_embedding_dim(
            embedding_dim=embedding_dim,
            force_2d=force_2d,
        )
    if force_2d and len(principal_components) != 2:
        raise ValueError(
            "--force-2d cannot be combined with a 3-component "
            "--principal-components selection"
        )
    return principal_components


def subplot_type_for_principal_components(principal_components: Sequence[int]) -> str:
    return "scene" if len(principal_components) == 3 else "xy"


def build_subplot_specs(
    keys: list[int],
    principal_components_by_key: dict[int, tuple[int, ...]],
    rows: int,
    cols: int,
) -> list[list[dict[str, str] | None]]:
    specs: list[list[dict[str, str] | None]] = []
    for row_idx in range(rows):
        row_specs = []
        for col_idx in range(cols):
            plot_idx = row_idx * cols + col_idx
            if plot_idx >= len(keys):
                row_specs.append(None)
                continue
            key = keys[plot_idx]
            row_specs.append(
                {
                    "type": subplot_type_for_principal_components(
                        principal_components_by_key[key]
                    )
                }
            )
        specs.append(row_specs)
    return specs


def subplot_grid(n_plots: int) -> tuple[int, int]:
    if n_plots <= 0:
        return 1, 1
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)
    return rows, cols


def build_role_figure(
    records: list[ProbeEmbeddingRecord],
    exclude_center_squares: bool = False,
    force_2d: bool = False,
    principal_components: tuple[int, ...] | None = None,
    standardize: bool = False,
    show_neighbor_lines: bool = False,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_tpr_embedding_pca.py. Install plotly first."
        ) from exc

    role_dims = sorted({record.role_dim for record in records})
    rows, cols = subplot_grid(len(role_dims))
    principal_components_by_role_dim = {
        role_dim: resolve_principal_components_for_embedding_dim(
            next(
                record.role_embeddings.shape[1]
                for record in records
                if record.role_dim == role_dim
            ),
            principal_components=principal_components,
            force_2d=force_2d,
        )
        for role_dim in role_dims
    }
    figure = make_subplots(
        rows=rows,
        cols=cols,
        specs=build_subplot_specs(
            role_dims,
            principal_components_by_role_dim,
            rows,
            cols,
        ),
        subplot_titles=[f"Role PCA: r={role_dim}" for role_dim in role_dims],
    )

    for idx, role_dim in enumerate(role_dims, start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        group_records = [record for record in records if record.role_dim == role_dim]
        selected_components = principal_components_by_role_dim[role_dim]
        n_components = max(selected_components)

        points = []
        point_metadata = []
        for record_idx, record in enumerate(group_records):
            for square_idx, embedding in enumerate(record.role_embeddings):
                if exclude_center_squares and square_idx in STARTING_SQUARES:
                    continue
                row_idx, col_idx = square_row_col(square_idx)
                points.append(embedding)
                point_metadata.append((record_idx, record, square_idx, row_idx, col_idx))

        if not points:
            raise ValueError("No role embedding points remain after applying filters")

        projection, explained_ratio = compute_pca(
            np.asarray(points, dtype=np.float32),
            n_components=n_components,
            standardize=standardize,
        )
        selected_projection = projection[:, [component - 1 for component in selected_components]]
        points_by_record_and_square = {
            record_idx: {} for record_idx in range(len(group_records))
        }

        row_buckets = {
            row_idx: {"coords": [], "colors": [], "hover_text": []}
            for row_idx in range(BOARD_ROWS)
        }
        for projected_point, (record_idx, record, square_idx, row_idx, col_idx) in zip(
            selected_projection, point_metadata
        ):
            points_by_record_and_square[record_idx][square_idx] = projected_point
            row_buckets[row_idx]["coords"].append(projected_point)
            row_buckets[row_idx]["colors"].append(row_column_color(row_idx, col_idx))
            row_buckets[row_idx]["hover_text"].append(
                "<br>".join(
                    [
                        f"square={square_label(square_idx)}",
                        f"row={ROW_LABELS[row_idx]}",
                        f"column={col_idx + 1}",
                        f"layer={record.layer}",
                        f"role_dim={record.role_dim}",
                        f"filler_dim={record.filler_dim}",
                        f"use_bias={record.use_bias}",
                        f"probe={record.probe_path}",
                    ]
                )
            )

        if show_neighbor_lines:
            add_board_neighbor_line_trace(
                figure,
                go,
                subplot_row=row,
                subplot_col=col,
                points_by_record_and_square=points_by_record_and_square,
                n_plot_components=len(selected_components),
                excluded_square_indices=STARTING_SQUARES if exclude_center_squares else None,
            )

        for row_idx in range(BOARD_ROWS):
            row_points = row_buckets[row_idx]["coords"]
            if not row_points:
                continue

            row_points_array = np.asarray(row_points, dtype=np.float64)
            marker = {
                "size": 7 if len(selected_components) == 3 else 8,
                "opacity": 0.82,
                "symbol": "circle",
                "color": row_buckets[row_idx]["colors"],
                "line": {"width": 0.5, "color": "rgba(255, 255, 255, 0.65)"},
            }
            trace_kwargs = {
                "x": row_points_array[:, 0],
                "y": row_points_array[:, 1],
                "mode": "markers",
                "marker": marker,
                "text": row_buckets[row_idx]["hover_text"],
                "hovertemplate": "%{text}<extra></extra>",
                "name": f"Row {ROW_LABELS[row_idx]}",
                "legendgroup": f"row-{row_idx}",
                "showlegend": False,
            }
            if len(selected_components) == 3:
                figure.add_trace(
                    go.Scatter3d(z=row_points_array[:, 2], **trace_kwargs),
                    row=row,
                    col=col,
                )
            else:
                figure.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)

        if len(selected_components) == 3:
            figure.update_scenes(
                xaxis_title_text=(
                    f"PC{selected_components[0]} "
                    f"({100 * explained_ratio[selected_components[0] - 1]:.1f}%)"
                ),
                yaxis_title_text=(
                    f"PC{selected_components[1]} "
                    f"({100 * explained_ratio[selected_components[1] - 1]:.1f}%)"
                ),
                zaxis_title_text=(
                    f"PC{selected_components[2]} "
                    f"({100 * explained_ratio[selected_components[2] - 1]:.1f}%)"
                ),
                row=row,
                col=col,
            )
        else:
            figure.update_xaxes(
                title_text=(
                    f"PC{selected_components[0]} "
                    f"({100 * explained_ratio[selected_components[0] - 1]:.1f}%)"
                ),
                row=row,
                col=col,
            )
            figure.update_yaxes(
                title_text=(
                    f"PC{selected_components[1]} "
                    f"({100 * explained_ratio[selected_components[1] - 1]:.1f}%)"
                ),
                row=row,
                col=col,
            )

    first_components = principal_components_by_role_dim[role_dims[0]]
    for row_idx in range(BOARD_ROWS):
        legend_marker = {
            "size": 9 if len(first_components) == 3 else 10,
            "opacity": 1.0,
            "symbol": "circle",
            "color": row_legend_color(row_idx),
            "line": {"width": 1.0, "color": "rgba(0, 0, 0, 0.55)"},
        }
        legend_kwargs = {
            "mode": "markers",
            "marker": legend_marker,
            "name": f"Row {ROW_LABELS[row_idx]}",
            "legendgroup": f"row-{row_idx}",
            "showlegend": True,
            "hoverinfo": "skip",
        }
        if len(first_components) == 3:
            figure.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    **legend_kwargs,
                ),
                row=1,
                col=1,
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    **legend_kwargs,
                ),
                row=1,
                col=1,
            )

    role_title = "TPR Role Embeddings: PCA (row=color family, column=shade)"
    if exclude_center_squares:
        role_title += " [center excluded]"
    if standardize:
        role_title += " [standardized]"
    if show_neighbor_lines:
        role_title += " [board neighbors]"
    figure.update_layout(
        title=role_title,
        height=max(450, 420 * rows),
        width=max(800, 420 * cols),
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    return figure


def build_filler_figure(
    records: list[ProbeEmbeddingRecord],
    force_2d: bool = False,
    principal_components: tuple[int, ...] | None = None,
    standardize: bool = False,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_tpr_embedding_pca.py. Install plotly first."
        ) from exc
    state_colors = {
        "empty": "#636EFA",
        "opponent": "#EF553B",
        "current": "#00CC96",
    }
    filler_dims = sorted({record.filler_dim for record in records})
    rows, cols = subplot_grid(len(filler_dims))
    principal_components_by_filler_dim = {
        filler_dim: resolve_principal_components_for_embedding_dim(
            next(
                record.filler_embeddings.shape[1]
                for record in records
                if record.filler_dim == filler_dim
            ),
            principal_components=principal_components,
            force_2d=force_2d,
        )
        for filler_dim in filler_dims
    }
    figure = make_subplots(
        rows=rows,
        cols=cols,
        specs=build_subplot_specs(
            filler_dims,
            principal_components_by_filler_dim,
            rows,
            cols,
        ),
        subplot_titles=[f"Filler PCA: f={filler_dim}" for filler_dim in filler_dims],
    )

    legend_shown = set()
    for idx, filler_dim in enumerate(filler_dims, start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        selected_components = principal_components_by_filler_dim[filler_dim]
        n_components = max(selected_components)
        group_records = [record for record in records if record.filler_dim == filler_dim]
        points = np.concatenate(
            [record.filler_embeddings for record in group_records],
            axis=0,
        )
        projection, explained_ratio = compute_pca(
            points,
            n_components=n_components,
            standardize=standardize,
        )
        selected_projection = projection[:, [component - 1 for component in selected_components]]

        for state_idx, state_label in enumerate(STATE_LABELS):
            state_rows = []
            hover_text = []
            for record_idx, record in enumerate(group_records):
                global_idx = record_idx * len(STATE_LABELS) + state_idx
                state_rows.append(selected_projection[global_idx])
                hover_text.append(
                    "<br>".join(
                        [
                            f"state={state_label}",
                            f"layer={record.layer}",
                            f"role_dim={record.role_dim}",
                            f"filler_dim={record.filler_dim}",
                            f"use_bias={record.use_bias}",
                            f"probe={record.probe_path}",
                        ]
                    )
                )
            state_points = np.asarray(state_rows, dtype=np.float64)
            trace_kwargs = {
                "x": state_points[:, 0],
                "y": state_points[:, 1],
                "mode": "markers",
                "marker": {
                    "size": 8 if len(selected_components) == 3 else 10,
                    "opacity": 0.82,
                    "color": state_colors[state_label],
                },
                "text": hover_text,
                "hovertemplate": "%{text}<extra></extra>",
                "name": state_label,
                "legendgroup": state_label,
                "showlegend": state_label not in legend_shown,
            }
            if len(selected_components) == 3:
                figure.add_trace(
                    go.Scatter3d(z=state_points[:, 2], **trace_kwargs),
                    row=row,
                    col=col,
                )
            else:
                figure.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
            legend_shown.add(state_label)

        if len(selected_components) == 3:
            figure.update_scenes(
                xaxis_title_text=(
                    f"PC{selected_components[0]} "
                    f"({100 * explained_ratio[selected_components[0] - 1]:.1f}%)"
                ),
                yaxis_title_text=(
                    f"PC{selected_components[1]} "
                    f"({100 * explained_ratio[selected_components[1] - 1]:.1f}%)"
                ),
                zaxis_title_text=(
                    f"PC{selected_components[2]} "
                    f"({100 * explained_ratio[selected_components[2] - 1]:.1f}%)"
                ),
                row=row,
                col=col,
            )
        else:
            figure.update_xaxes(
                title_text=(
                    f"PC{selected_components[0]} "
                    f"({100 * explained_ratio[selected_components[0] - 1]:.1f}%)"
                ),
                row=row,
                col=col,
            )
            figure.update_yaxes(
                title_text=(
                    f"PC{selected_components[1]} "
                    f"({100 * explained_ratio[selected_components[1] - 1]:.1f}%)"
                ),
                row=row,
                col=col,
            )

    filler_title = "TPR Filler Embeddings: PCA"
    if standardize:
        filler_title += " [standardized]"
    figure.update_layout(
        title=filler_title,
        height=max(450, 420 * rows),
        width=max(800, 420 * cols),
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    return figure


def write_html_report(
    role_figure,
    filler_figure,
    output_path: Path,
    records: list[ProbeEmbeddingRecord],
    exclude_center_squares: bool = False,
    force_2d: bool = False,
    principal_components: tuple[int, ...] | None = None,
    standardize: bool = False,
    show_neighbor_lines: bool = False,
) -> None:
    import plotly.io as pio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted({record.layer for record in records})
    role_dims = sorted({record.role_dim for record in records})
    filler_dims = sorted({record.filler_dim for record in records})
    role_plot_html = wrap_plot_html_with_camera_overlay(
        pio.to_html(role_figure, include_plotlyjs="cdn", full_html=False)
    )
    filler_plot_html = wrap_plot_html_with_camera_overlay(
        pio.to_html(filler_figure, include_plotlyjs=False, full_html=False)
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TPR Embedding PCA Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      line-height: 1.45;
    }}
    h1, h2 {{
      margin-bottom: 0.35em;
    }}
    .meta {{
      margin-bottom: 20px;
      color: #333;
    }}
    .plot {{
      margin-top: 24px;
      margin-bottom: 24px;
    }}
    code {{
      background: #f2f4f8;
      padding: 0.1em 0.35em;
      border-radius: 4px;
    }}
{camera_overlay_style_block()}
  </style>
</head>
<body>
  <h1>TPR Embedding PCA Report</h1>
  <div class="meta">
    <div>Loaded <code>{len(records)}</code> TPR probe checkpoints.</div>
    <div>Layers: <code>{",".join(str(value) for value in layers)}</code></div>
    <div>Role dims: <code>{",".join(str(value) for value in role_dims)}</code></div>
    <div>Filler dims: <code>{",".join(str(value) for value in filler_dims)}</code></div>
    <div>Role view excludes center squares: <code>{str(exclude_center_squares).lower()}</code></div>
    <div>Forced 2D PCA: <code>{str(force_2d).lower()}</code></div>
    <div>Principal components: <code>{",".join(str(value) for value in principal_components) if principal_components is not None else "default"}</code></div>
    <div>StandardScaler preprocessing: <code>{str(standardize).lower()}</code></div>
    <div>Board-neighbor lines on role plots: <code>{str(show_neighbor_lines).lower()}</code></div>
  </div>
  <div class="plot">
    <h2>Role Embeddings</h2>
    {role_plot_html}
  </div>
  <div class="plot">
    <h2>Filler Embeddings</h2>
    {filler_plot_html}
  </div>
{camera_overlay_script_block()}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def build_pdf_output_paths(output_path: Path) -> tuple[Path, Path]:
    if output_path.suffix:
        base_path = output_path.with_suffix("")
    else:
        base_path = output_path
    role_pdf_path = base_path.parent / f"{base_path.name}_role.pdf"
    filler_pdf_path = base_path.parent / f"{base_path.name}_filler.pdf"
    return role_pdf_path, filler_pdf_path


def build_html_output_path(output_path: Path) -> Path:
    if output_path.suffix == ".html":
        return output_path
    if output_path.suffix:
        base_path = output_path.with_suffix("")
    else:
        base_path = output_path
    return base_path.parent / f"{base_path.name}.html"


def write_pdf_exports(role_figure, filler_figure, output_path: Path) -> tuple[Path, Path]:
    import plotly.io as pio

    role_pdf_path, filler_pdf_path = build_pdf_output_paths(output_path)
    role_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    filler_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    kaleido_scope = getattr(getattr(pio, "kaleido", None), "scope", None)
    original_mathjax = None
    if kaleido_scope is not None and hasattr(kaleido_scope, "mathjax"):
        original_mathjax = kaleido_scope.mathjax
        kaleido_scope.mathjax = None
    try:
        pio.write_image(role_figure, role_pdf_path, format="pdf")
        pio.write_image(filler_figure, filler_pdf_path, format="pdf")
    except Exception as exc:
        raise RuntimeError(
            "Failed to export PCA figures as PDFs. Ensure plotly static image "
            "export dependencies such as kaleido are installed."
        ) from exc
    finally:
        if kaleido_scope is not None and hasattr(kaleido_scope, "mathjax"):
            kaleido_scope.mathjax = original_mathjax
    return role_pdf_path, filler_pdf_path


def wrap_plot_html_with_camera_overlay(plot_html: str) -> str:
    return f'<div class="camera-overlay-root">{plot_html}</div>'


def camera_overlay_style_block() -> str:
    return """
  .camera-overlay-root {
    position: relative;
  }
  .camera-overlay-layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 20;
  }
  .camera-overlay-box {
    position: absolute;
    max-width: 32ch;
    padding: 6px 8px;
    border-radius: 6px;
    border: 1px solid rgba(0, 0, 0, 0.18);
    background: rgba(255, 255, 255, 0.86);
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.12);
    color: #111;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    line-height: 1.35;
    white-space: pre-wrap;
  }
  .camera-overlay-label {
    display: block;
    margin-bottom: 2px;
    font-weight: 600;
  }
"""


def camera_overlay_script_block() -> str:
    return """
  <script>
    (function () {
      function formatNumber(value) {
        const numericValue = Number(value);
        if (!Number.isFinite(numericValue)) {
          return "null";
        }
        return numericValue.toFixed(3);
      }

      function sceneOrder(sceneKey) {
        if (sceneKey === "scene") {
          return 1;
        }
        const suffix = Number(sceneKey.slice("scene".length));
        return Number.isFinite(suffix) ? suffix : Number.MAX_SAFE_INTEGER;
      }

      function formatCameraText(camera) {
        const safeCamera = camera || {};
        const eye = safeCamera.eye || {};
        const center = safeCamera.center || {};
        const up = safeCamera.up || {};
        return [
          "eye=(" + formatNumber(eye.x) + ", " + formatNumber(eye.y) + ", " + formatNumber(eye.z) + ")",
          "center=(" + formatNumber(center.x) + ", " + formatNumber(center.y) + ", " + formatNumber(center.z) + ")",
          "up=(" + formatNumber(up.x) + ", " + formatNumber(up.y) + ", " + formatNumber(up.z) + ")",
        ].join("\\n");
      }

      function sceneKeys(layout) {
        return Object.keys(layout)
          .filter(function (key) {
            return key.startsWith("scene") && layout[key] && typeof layout[key] === "object" && layout[key].domain;
          })
          .sort(function (left, right) {
            return sceneOrder(left) - sceneOrder(right);
          });
      }

      function ensureOverlayLayer(root) {
        let overlayLayer = root.querySelector(".camera-overlay-layer");
        if (!overlayLayer) {
          overlayLayer = document.createElement("div");
          overlayLayer.className = "camera-overlay-layer";
          root.appendChild(overlayLayer);
        }
        return overlayLayer;
      }

      function renderCameraOverlays(graphDiv) {
        if (!graphDiv || !graphDiv._fullLayout) {
          return;
        }
        const root = graphDiv.closest(".camera-overlay-root");
        if (!root) {
          return;
        }
        const overlayLayer = ensureOverlayLayer(root);
        overlayLayer.replaceChildren();

        sceneKeys(graphDiv._fullLayout).forEach(function (sceneKey) {
          const scene = graphDiv._fullLayout[sceneKey];
          if (!scene || !scene.domain) {
            return;
          }
          const overlayBox = document.createElement("div");
          overlayBox.className = "camera-overlay-box";
          overlayBox.style.left = "calc(" + (100 * scene.domain.x[0]).toFixed(2) + "% + 8px)";
          overlayBox.style.top = "calc(" + (100 * (1 - scene.domain.y[1])).toFixed(2) + "% + 8px)";
          overlayBox.style.maxWidth = "calc(" + (100 * (scene.domain.x[1] - scene.domain.x[0])).toFixed(2) + "% - 16px)";

          const label = document.createElement("span");
          label.className = "camera-overlay-label";
          label.textContent = sceneKey + ".camera";

          const body = document.createElement("span");
          body.textContent = formatCameraText(scene.camera);

          overlayBox.appendChild(label);
          overlayBox.appendChild(body);
          overlayLayer.appendChild(overlayBox);
        });
      }

      function installCameraOverlay(graphDiv) {
        if (!graphDiv || graphDiv.__cameraOverlayInstalled) {
          return;
        }
        graphDiv.__cameraOverlayInstalled = true;
        const rerender = function () {
          renderCameraOverlays(graphDiv);
        };
        window.addEventListener("resize", rerender);
        let handlersRegistered = false;

        function registerHandlers() {
          if (handlersRegistered || typeof graphDiv.on !== "function") {
            return handlersRegistered;
          }
          graphDiv.on("plotly_relayout", rerender);
          graphDiv.on("plotly_afterplot", rerender);
          handlersRegistered = true;
          return true;
        }

        registerHandlers();

        let attempts = 0;
        const maxAttempts = 50;
        const waitForPlot = window.setInterval(function () {
          attempts += 1;
          registerHandlers();
          if (graphDiv._fullLayout) {
            window.clearInterval(waitForPlot);
            rerender();
          } else if (attempts >= maxAttempts) {
            window.clearInterval(waitForPlot);
          }
        }, 200);
      }

      function installAllCameraOverlays() {
        document
          .querySelectorAll(".camera-overlay-root .plotly-graph-div")
          .forEach(installCameraOverlay);
      }

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
          window.setTimeout(installAllCameraOverlays, 0);
        });
      } else {
        window.setTimeout(installAllCameraOverlays, 0);
      }
      window.addEventListener("load", installAllCameraOverlays);
    })();
  </script>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "probe_paths",
        nargs="*",
        metavar="PATH",
        help=(
            "Direct TPR checkpoint file paths or directories. Directories are "
            "searched recursively for resid_*_tpr_*.pth."
        ),
    )
    parser.add_argument(
        "--input-paths",
        nargs="+",
        metavar="PATH",
        dest="legacy_input_paths",
        help=(
            "Legacy form of the input path list. Accepts the same checkpoint "
            "files or directories as the positional PATH arguments."
        ),
    )
    parser.add_argument(
        "--layers",
        help="Optional comma-separated layer filter.",
    )
    parser.add_argument(
        "--role-dims",
        help="Optional comma-separated role-dimension filter.",
    )
    parser.add_argument(
        "--filler-dims",
        help="Optional comma-separated filler-dimension filter.",
    )
    parser.add_argument(
        "--include-bias",
        action="store_true",
        help="Include bias-enabled TPR checkpoints. By default they are skipped.",
    )
    parser.add_argument(
        "--exclude-center-squares",
        action="store_true",
        help="Exclude the four center starting squares from the role-embedding PCA view.",
    )
    parser.add_argument(
        "--force-2d",
        action="store_true",
        help="Force every subplot to use 2D PCA even when the embedding dimension allows 3D.",
    )
    parser.add_argument(
        "--principal-components",
        help=(
            "Comma-separated 1-based principal-component indices to plot, such as "
            "`1,2`, `1,3`, or `2,3,4`. Use exactly 2 values for 2D or 3 values "
            "for 3D. By default the script uses the leading PCs implied by the "
            "embedding dimension and --force-2d."
        ),
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply sklearn.preprocessing.StandardScaler to each embedding point cloud before PCA.",
    )
    parser.add_argument(
        "--show-neighbor-lines",
        action="store_true",
        help="Draw faint line segments between role-embedding points for neighboring board squares.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help=(
            "Shared output base. The script writes the interactive HTML report "
            "to <base>.html unless the path already ends in .html, and writes "
            "the PDF exports to <base>_role.pdf and <base>_filler.pdf."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_paths = list(args.probe_paths)
    if args.legacy_input_paths is not None:
        input_paths.extend(args.legacy_input_paths)
    if not input_paths:
        parser.error("provide at least one TPR checkpoint path or directory")
    try:
        principal_components = parse_principal_components(args.principal_components)
    except ValueError as exc:
        parser.error(str(exc))

    probe_paths = discover_probe_paths(
        input_paths=input_paths,
        layers=parse_int_list(args.layers),
        role_dims=parse_int_list(args.role_dims),
        filler_dims=parse_int_list(args.filler_dims),
        include_bias=args.include_bias,
    )
    records = load_probe_embedding_records(probe_paths)
    role_figure = build_role_figure(
        records,
        exclude_center_squares=args.exclude_center_squares,
        force_2d=args.force_2d,
        principal_components=principal_components,
        standardize=args.standardize,
        show_neighbor_lines=args.show_neighbor_lines,
    )
    filler_figure = build_filler_figure(
        records,
        force_2d=args.force_2d,
        principal_components=principal_components,
        standardize=args.standardize,
    )
    output_path = Path(args.output_path).expanduser()
    html_output_path = build_html_output_path(output_path)
    write_html_report(
        role_figure=role_figure,
        filler_figure=filler_figure,
        output_path=html_output_path,
        records=records,
        exclude_center_squares=args.exclude_center_squares,
        force_2d=args.force_2d,
        principal_components=principal_components,
        standardize=args.standardize,
        show_neighbor_lines=args.show_neighbor_lines,
    )
    role_pdf_path, filler_pdf_path = write_pdf_exports(
        role_figure=role_figure,
        filler_figure=filler_figure,
        output_path=output_path,
    )
    print(f"Wrote interactive PCA report to {html_output_path}")
    print(f"Wrote role PCA PDF to {role_pdf_path}")
    print(f"Wrote filler PCA PDF to {filler_pdf_path}")


if __name__ == "__main__":
    main()
