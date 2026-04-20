"""Shared helpers for multilinear TPR intervention scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from intervene_probe import probe_patch_channels_for_square_color
from intervene_tpr_probe import parse_explicit_probe_pairs  # noqa: E402
from train_multilinear_tpr_probe import load_saved_multilinear_tpr_probe  # noqa: E402


def make_multilinear_probe_stem(
    layer: int,
    row_dim: int,
    col_dim: int,
    color_dim: int,
    use_bias: bool,
    exclude_center_squares: bool,
) -> str:
    stem = f"resid_{layer}_mltpr_row{row_dim}_col{col_dim}_color{color_dim}"
    if use_bias:
        stem += "_bias"
    if exclude_center_squares:
        stem += "_no_center"
    return stem


def resolve_multilinear_probe_path(
    probe_dir: str | Path,
    layer: int,
    row_dim: int,
    col_dim: int,
    color_dim: int,
    use_bias: bool,
    exclude_center_squares: bool,
    probe_seed: int | None = None,
) -> Path:
    probe_dir = Path(probe_dir)
    stem = make_multilinear_probe_stem(
        layer=layer,
        row_dim=row_dim,
        col_dim=col_dim,
        color_dim=color_dim,
        use_bias=use_bias,
        exclude_center_squares=exclude_center_squares,
    )

    preferred_names = []
    if probe_seed is not None:
        preferred_names.append(f"{stem}_seed{probe_seed}.pth")
    preferred_names.append(f"{stem}.pth")
    if probe_seed is None:
        search_patterns = [f"{stem}_seed*.pth", f"{stem}.pth"]
    else:
        search_patterns = [f"{stem}_seed{probe_seed}.pth", f"{stem}.pth"]

    for filename in preferred_names:
        direct_path = probe_dir / filename
        if direct_path.exists():
            return direct_path

    matches: list[Path] = []
    seen_paths: set[str] = set()
    for pattern in search_patterns:
        for match in sorted(probe_dir.rglob(pattern)):
            resolved = str(match.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            matches.append(match)

    if not matches:
        hint = (
            f"{stem}_seed{probe_seed}.pth"
            if probe_seed is not None
            else f"{stem}_seed*.pth"
        )
        raise FileNotFoundError(f"Could not find {hint} under {probe_dir}")
    if len(matches) > 1:
        joined_matches = ", ".join(str(match) for match in matches)
        raise FileNotFoundError(
            f"Found multiple matches for {stem} under {probe_dir}: {joined_matches}. "
            "Use --probe-seed, --probe-path, or explicit --probe-pair paths to disambiguate."
        )
    return matches[0]


def load_multilinear_factors(
    probe_path: str | Path,
    d_model: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    dict,
]:
    probe, layer, artifact = load_saved_multilinear_tpr_probe(
        probe_path=probe_path,
        d_model=d_model,
        device=device,
    )
    return (
        probe.binding_map.detach().to(device),
        probe.row_embeddings.detach().to(device),
        probe.col_embeddings.detach().to(device),
        probe.color_embeddings.detach().to(device),
        layer,
        artifact,
    )


def artifact_has_binding_to_residual(artifact: dict) -> bool:
    saved_map = artifact.get("binding_to_residual")
    return isinstance(saved_map, dict) and isinstance(
        saved_map.get("weight"), torch.Tensor
    )


def build_multilinear_binding_space_constraints(
    row_embeddings: torch.Tensor,
    col_embeddings: torch.Tensor,
    color_embeddings: torch.Tensor,
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    move_idx: int,
    intervention_type: str = "flip",
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(pos_ints) != len(ori_colors):
        raise ValueError("pos_ints and ori_colors must have the same length")

    selected_spatial_features = []
    selected_color_deltas = []
    for pos_int, ori_color in zip(pos_ints, ori_colors, strict=True):
        row_idx, col_idx = divmod(int(pos_int), 8)
        source_label, target_label = probe_patch_channels_for_square_color(
            original_color=int(ori_color),
            move_idx=move_idx,
            intervention_type=intervention_type,
        )
        spatial_feature = torch.einsum(
            "r,c->rc",
            row_embeddings[row_idx, :],
            col_embeddings[col_idx, :],
        ).reshape(-1)
        selected_spatial_features.append(spatial_feature)
        selected_color_deltas.append(
            color_embeddings[target_label] - color_embeddings[source_label]
        )

    return (
        torch.stack(selected_spatial_features, dim=0),
        torch.stack(selected_color_deltas, dim=0),
    )


def build_multilinear_binding_delta_as_sum_of_outer_products(
    selected_spatial_features: torch.Tensor,
    selected_color_deltas: torch.Tensor,
    *,
    row_dim: int,
    col_dim: int,
) -> torch.Tensor:
    delta_binding_flat = torch.einsum(
        "is,ik->sk",
        selected_spatial_features,
        selected_color_deltas,
    )
    return delta_binding_flat.reshape(row_dim, col_dim, selected_color_deltas.shape[-1])


def solve_residual_delta_for_binding_delta(
    binding_map: torch.Tensor,
    delta_binding: torch.Tensor,
) -> torch.Tensor:
    delta_binding_flat = delta_binding.reshape(-1)
    binding_map_flat = binding_map.reshape(binding_map.shape[0], -1)
    return torch.linalg.pinv(binding_map_flat.T) @ delta_binding_flat


def resolve_multilinear_resources_for_patch_layers(
    *,
    probe_pairs: tuple[str, ...],
    probe_seed: int | None,
    patch_layers: Sequence[int],
    probe_dir: str | Path,
    row_dim: int,
    col_dim: int,
    color_dim: int,
    use_bias: bool,
    exclude_center_squares: bool,
    d_model: int,
    device: torch.device,
) -> tuple[
    list[int],
    dict[int, int],
    dict[int, str],
    dict[int, dict],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    explicit_probe_pairs = parse_explicit_probe_pairs(probe_pairs)
    resolved_patch_layers = (
        sorted(explicit_probe_pairs) if explicit_probe_pairs else list(patch_layers)
    )

    probe_source_layers: dict[int, int] = {}
    probe_paths_by_patch_layer: dict[int, str] = {}
    probe_configs_by_patch_layer: dict[int, dict] = {}
    binding_maps_by_patch_layer: dict[int, torch.Tensor] = {}
    row_embeddings_by_patch_layer: dict[int, torch.Tensor] = {}
    col_embeddings_by_patch_layer: dict[int, torch.Tensor] = {}
    color_embeddings_by_patch_layer: dict[int, torch.Tensor] = {}

    if explicit_probe_pairs:
        for patch_layer in resolved_patch_layers:
            current_probe_path = explicit_probe_pairs[patch_layer]
            (
                binding_map,
                row_embeddings,
                col_embeddings,
                color_embeddings,
                loaded_layer,
                artifact,
            ) = load_multilinear_factors(
                probe_path=current_probe_path,
                d_model=d_model,
                device=device,
            )
            probe_source_layers[patch_layer] = loaded_layer
            probe_paths_by_patch_layer[patch_layer] = str(current_probe_path)
            probe_configs_by_patch_layer[patch_layer] = {
                "layer": loaded_layer,
                "row_dim": int(artifact["row_dim"]),
                "col_dim": int(artifact["col_dim"]),
                "color_dim": int(artifact["color_dim"]),
                "use_bias": bool(artifact.get("use_bias", False)),
                "exclude_center_squares": bool(
                    artifact.get("config", {}).get("exclude_center_squares", False)
                ),
                "has_binding_to_residual": artifact_has_binding_to_residual(artifact),
            }
            binding_maps_by_patch_layer[patch_layer] = binding_map
            row_embeddings_by_patch_layer[patch_layer] = row_embeddings
            col_embeddings_by_patch_layer[patch_layer] = col_embeddings
            color_embeddings_by_patch_layer[patch_layer] = color_embeddings
    else:
        binding_maps_by_source_layer: dict[int, torch.Tensor] = {}
        row_embeddings_by_source_layer: dict[int, torch.Tensor] = {}
        col_embeddings_by_source_layer: dict[int, torch.Tensor] = {}
        color_embeddings_by_source_layer: dict[int, torch.Tensor] = {}
        probe_paths_by_source_layer: dict[int, str] = {}
        probe_configs_by_source_layer: dict[int, dict] = {}

        for source_layer in resolved_patch_layers:
            current_probe_path = resolve_multilinear_probe_path(
                probe_dir=probe_dir,
                layer=source_layer,
                row_dim=row_dim,
                col_dim=col_dim,
                color_dim=color_dim,
                use_bias=use_bias,
                exclude_center_squares=exclude_center_squares,
                probe_seed=probe_seed,
            )
            (
                binding_map,
                row_embeddings,
                col_embeddings,
                color_embeddings,
                loaded_layer,
                artifact,
            ) = load_multilinear_factors(
                probe_path=current_probe_path,
                d_model=d_model,
                device=device,
            )
            if loaded_layer != source_layer:
                raise ValueError(
                    f"Loaded multilinear TPR checkpoint layer {loaded_layer} from "
                    f"{current_probe_path}, expected layer {source_layer}"
                )
            binding_maps_by_source_layer[source_layer] = binding_map
            row_embeddings_by_source_layer[source_layer] = row_embeddings
            col_embeddings_by_source_layer[source_layer] = col_embeddings
            color_embeddings_by_source_layer[source_layer] = color_embeddings
            probe_paths_by_source_layer[source_layer] = str(current_probe_path)
            probe_configs_by_source_layer[source_layer] = {
                "layer": loaded_layer,
                "row_dim": int(artifact["row_dim"]),
                "col_dim": int(artifact["col_dim"]),
                "color_dim": int(artifact["color_dim"]),
                "use_bias": bool(artifact.get("use_bias", False)),
                "exclude_center_squares": bool(
                    artifact.get("config", {}).get("exclude_center_squares", False)
                ),
                "has_binding_to_residual": artifact_has_binding_to_residual(artifact),
            }

        probe_source_layers = {
            layer: layer for layer in resolved_patch_layers
        }
        binding_maps_by_patch_layer = {
            layer: binding_maps_by_source_layer[probe_source_layers[layer]]
            for layer in resolved_patch_layers
        }
        row_embeddings_by_patch_layer = {
            layer: row_embeddings_by_source_layer[probe_source_layers[layer]]
            for layer in resolved_patch_layers
        }
        col_embeddings_by_patch_layer = {
            layer: col_embeddings_by_source_layer[probe_source_layers[layer]]
            for layer in resolved_patch_layers
        }
        color_embeddings_by_patch_layer = {
            layer: color_embeddings_by_source_layer[probe_source_layers[layer]]
            for layer in resolved_patch_layers
        }
        probe_paths_by_patch_layer = {
            layer: probe_paths_by_source_layer[probe_source_layers[layer]]
            for layer in resolved_patch_layers
        }
        probe_configs_by_patch_layer = {
            layer: probe_configs_by_source_layer[probe_source_layers[layer]]
            for layer in resolved_patch_layers
        }

    return (
        resolved_patch_layers,
        probe_source_layers,
        probe_paths_by_patch_layer,
        probe_configs_by_patch_layer,
        binding_maps_by_patch_layer,
        row_embeddings_by_patch_layer,
        col_embeddings_by_patch_layer,
        color_embeddings_by_patch_layer,
    )


def shard_benchmark_samples(
    samples: Sequence[dict],
    *,
    num_benchmark_shards: int,
    benchmark_shard_index: int,
) -> list[dict]:
    if num_benchmark_shards <= 0:
        raise ValueError("--num-benchmark-shards must be positive")
    if benchmark_shard_index < 0 or benchmark_shard_index >= num_benchmark_shards:
        raise ValueError(
            "--benchmark-shard-index must satisfy "
            f"0 <= index < {num_benchmark_shards}, got {benchmark_shard_index}"
        )
    if num_benchmark_shards == 1:
        return list(samples)
    return [
        sample
        for sample_index, sample in enumerate(samples)
        if sample_index % num_benchmark_shards == benchmark_shard_index
    ]
