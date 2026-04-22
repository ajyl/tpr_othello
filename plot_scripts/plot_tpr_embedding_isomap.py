"""Create interactive Isomap plots for TPR role and filler embeddings."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_tpr_embedding_pca import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    ProbeEmbeddingRecord,
    ROW_LABELS,
    STARTING_SQUARES,
    STATE_LABELS,
    add_board_neighbor_line_trace,
    build_html_output_path,
    camera_overlay_script_block,
    camera_overlay_style_block,
    discover_probe_paths,
    load_probe_embedding_records,
    maybe_standardize_points,
    parse_int_list,
    row_legend_color,
    row_column_color,
    square_label,
    square_row_col,
    wrap_plot_html_with_camera_overlay,
    write_pdf_exports,
)


DEFAULT_N_COMPONENTS = 3
DEFAULT_N_NEIGHBORS = 16
DEFAULT_METRIC = "euclidean"


def subplot_type_for_components(n_components: int) -> str:
    return "scene" if n_components == 3 else "xy"


def build_subplot_specs(
    keys: list[int],
    n_components: int,
    rows: int,
    cols: int,
) -> list[list[dict[str, str] | None]]:
    specs: list[list[dict[str, str] | None]] = []
    subplot_type = subplot_type_for_components(n_components)
    for row_idx in range(rows):
        row_specs = []
        for col_idx in range(cols):
            plot_idx = row_idx * cols + col_idx
            if plot_idx >= len(keys):
                row_specs.append(None)
            else:
                row_specs.append({"type": subplot_type})
        specs.append(row_specs)
    return specs


def subplot_grid(n_plots: int) -> tuple[int, int]:
    if n_plots <= 0:
        return 1, 1
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)
    return rows, cols


def pad_projection(projection: np.ndarray, n_components: int) -> np.ndarray:
    if projection.shape[1] >= n_components:
        return projection[:, :n_components]
    return np.pad(
        projection,
        ((0, 0), (0, n_components - projection.shape[1])),
        mode="constant",
        constant_values=0.0,
    )


def compute_isomap_projection(
    points: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int,
    metric: str,
    standardize: bool,
) -> np.ndarray:
    try:
        from sklearn.manifold import Isomap
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for plot_tpr_embedding_isomap.py. Install scikit-learn first."
        ) from exc

    points = maybe_standardize_points(points, standardize=standardize)
    if points.shape[0] < 2:
        raise ValueError("Isomap requires at least two points")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    effective_n_neighbors = min(max(1, n_neighbors), max(1, points.shape[0] - 1))
    effective_n_components = min(
        n_components,
        max(1, min(points.shape[0] - 1, points.shape[1])),
    )
    reducer = Isomap(
        n_components=effective_n_components,
        n_neighbors=effective_n_neighbors,
        metric=metric,
    )
    projection = reducer.fit_transform(points)
    return pad_projection(projection, n_components)


def build_role_figure(
    records: list[ProbeEmbeddingRecord],
    *,
    exclude_center_squares: bool,
    n_components: int,
    n_neighbors: int,
    metric: str,
    standardize: bool,
    show_neighbor_lines: bool,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_tpr_embedding_isomap.py. Install plotly first."
        ) from exc

    role_dims = sorted({record.role_dim for record in records})
    rows, cols = subplot_grid(len(role_dims))
    figure = make_subplots(
        rows=rows,
        cols=cols,
        specs=build_subplot_specs(role_dims, n_components, rows, cols),
        subplot_titles=[f"Role Isomap: r={role_dim}" for role_dim in role_dims],
    )

    for idx, role_dim in enumerate(role_dims, start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        group_records = [record for record in records if record.role_dim == role_dim]

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

        projection = compute_isomap_projection(
            np.asarray(points, dtype=np.float32),
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            standardize=standardize,
        )
        points_by_record_and_square = {
            record_idx: {} for record_idx in range(len(group_records))
        }

        row_buckets = {
            row_idx: {"coords": [], "colors": [], "hover_text": []}
            for row_idx in range(BOARD_ROWS)
        }
        for projected_point, (record_idx, record, square_idx, row_idx, col_idx) in zip(
            projection, point_metadata
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
                n_plot_components=n_components,
                excluded_square_indices=STARTING_SQUARES if exclude_center_squares else None,
            )

        for row_idx in range(BOARD_ROWS):
            row_points = row_buckets[row_idx]["coords"]
            if not row_points:
                continue

            row_points_array = np.asarray(row_points, dtype=np.float64)
            marker = {
                "size": 7 if n_components == 3 else 8,
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
            if n_components == 3:
                figure.add_trace(
                    go.Scatter3d(z=row_points_array[:, 2], **trace_kwargs),
                    row=row,
                    col=col,
                )
            else:
                figure.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)

        if n_components == 3:
            figure.update_scenes(
                xaxis_title_text="Isomap1",
                yaxis_title_text="Isomap2",
                zaxis_title_text="Isomap3",
                row=row,
                col=col,
            )
        else:
            figure.update_xaxes(title_text="Isomap1", row=row, col=col)
            figure.update_yaxes(title_text="Isomap2", row=row, col=col)

    for row_idx in range(BOARD_ROWS):
        legend_marker = {
            "size": 9 if n_components == 3 else 10,
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
        if n_components == 3:
            figure.add_trace(
                go.Scatter3d(x=[None], y=[None], z=[None], **legend_kwargs),
                row=1,
                col=1,
            )
        else:
            figure.add_trace(
                go.Scatter(x=[None], y=[None], **legend_kwargs),
                row=1,
                col=1,
            )

    role_title = (
        "TPR Role Embeddings: Isomap "
        f"(n_neighbors={n_neighbors}, metric={metric}) "
        "[row=color family, column=shade]"
    )
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
    *,
    n_components: int,
    n_neighbors: int,
    metric: str,
    standardize: bool,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_tpr_embedding_isomap.py. Install plotly first."
        ) from exc

    state_colors = {
        "empty": "#636EFA",
        "opponent": "#EF553B",
        "current": "#00CC96",
    }
    filler_dims = sorted({record.filler_dim for record in records})
    rows, cols = subplot_grid(len(filler_dims))
    figure = make_subplots(
        rows=rows,
        cols=cols,
        specs=build_subplot_specs(filler_dims, n_components, rows, cols),
        subplot_titles=[f"Filler Isomap: f={filler_dim}" for filler_dim in filler_dims],
    )

    legend_shown = set()
    for idx, filler_dim in enumerate(filler_dims, start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        group_records = [record for record in records if record.filler_dim == filler_dim]
        points = np.concatenate(
            [record.filler_embeddings for record in group_records],
            axis=0,
        )
        projection = compute_isomap_projection(
            points,
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            standardize=standardize,
        )

        for state_idx, state_label in enumerate(STATE_LABELS):
            state_rows = []
            hover_text = []
            for record_idx, record in enumerate(group_records):
                global_idx = record_idx * len(STATE_LABELS) + state_idx
                state_rows.append(projection[global_idx])
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
                    "size": 8 if n_components == 3 else 10,
                    "opacity": 0.82,
                    "color": state_colors[state_label],
                },
                "text": hover_text,
                "hovertemplate": "%{text}<extra></extra>",
                "name": state_label,
                "legendgroup": state_label,
                "showlegend": state_label not in legend_shown,
            }
            if n_components == 3:
                figure.add_trace(
                    go.Scatter3d(z=state_points[:, 2], **trace_kwargs),
                    row=row,
                    col=col,
                )
            else:
                figure.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
            legend_shown.add(state_label)

        if n_components == 3:
            figure.update_scenes(
                xaxis_title_text="Isomap1",
                yaxis_title_text="Isomap2",
                zaxis_title_text="Isomap3",
                row=row,
                col=col,
            )
        else:
            figure.update_xaxes(title_text="Isomap1", row=row, col=col)
            figure.update_yaxes(title_text="Isomap2", row=row, col=col)

    filler_title = (
        f"TPR Filler Embeddings: Isomap (n_neighbors={n_neighbors}, metric={metric})"
    )
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
    *,
    output_path: Path,
    records: list[ProbeEmbeddingRecord],
    exclude_center_squares: bool,
    n_components: int,
    n_neighbors: int,
    metric: str,
    standardize: bool,
    show_neighbor_lines: bool,
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
  <title>TPR Embedding Isomap Report</title>
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
  <h1>TPR Embedding Isomap Report</h1>
  <div class="meta">
    <div>Loaded <code>{len(records)}</code> TPR probe checkpoints.</div>
    <div>Layers: <code>{",".join(str(value) for value in layers)}</code></div>
    <div>Role dims: <code>{",".join(str(value) for value in role_dims)}</code></div>
    <div>Filler dims: <code>{",".join(str(value) for value in filler_dims)}</code></div>
    <div>Role view excludes center squares: <code>{str(exclude_center_squares).lower()}</code></div>
    <div>Isomap components: <code>{n_components}</code></div>
    <div>Isomap n_neighbors: <code>{n_neighbors}</code></div>
    <div>Isomap metric: <code>{metric}</code></div>
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
    parser.add_argument("--layers", help="Optional comma-separated layer filter.")
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
        help="Exclude the four center starting squares from the role-embedding Isomap view.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        choices=(2, 3),
        default=DEFAULT_N_COMPONENTS,
        help="Number of Isomap dimensions to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help="Isomap n_neighbors hyperparameter (default: %(default)s).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Isomap distance metric (default: %(default)s).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply sklearn.preprocessing.StandardScaler to each embedding point cloud before Isomap.",
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
    if args.n_neighbors < 1:
        parser.error("--n-neighbors must be at least 1")

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
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
        standardize=args.standardize,
        show_neighbor_lines=args.show_neighbor_lines,
    )
    filler_figure = build_filler_figure(
        records,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
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
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
        standardize=args.standardize,
        show_neighbor_lines=args.show_neighbor_lines,
    )
    role_pdf_path, filler_pdf_path = write_pdf_exports(
        role_figure=role_figure,
        filler_figure=filler_figure,
        output_path=output_path,
    )
    print(f"Wrote interactive Isomap report to {html_output_path}")
    print(f"Wrote role Isomap PDF to {role_pdf_path}")
    print(f"Wrote filler Isomap PDF to {filler_pdf_path}")


if __name__ == "__main__":
    main()
