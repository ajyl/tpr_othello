"""Run configurable 1-4 square TPR interventions on Othello GPT."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "hook_utils"))

from data_utils import load_pickle_sequences  # noqa: E402
from load_model import load_model  # noqa: E402
from hook_utils.intervene import intervene  # noqa: E402
from hook_utils.record_utils import convert_to_hooked_model  # noqa: E402
from hook_utils.util_funcs import seed_all  # noqa: E402
from intervene_probe import (  # noqa: E402
    ACTUAL_INTERVENTION_TYPE_CHOICES,
    EMPTY,
    INTERVENTION_TYPE_CHOICES,
    OthelloBoardState,
    STARTING_SQUARES,
    assign_intervention_types_to_samples,
    board_pos_to_label,
    color_code_to_label,
    compute_pre_and_post_valids_for_squares,
    compute_prediction_error,
    encode_game_as_model_tokens,
    format_move_comparison_table,
    format_move_prediction_with_probability,
    format_selected_predictions_alphanumerically,
    format_topk_predictions_alphanumerically,
    load_benchmark_artifact,
    make_additive_patch,
    normalize_benchmark_samples,
    normalize_square_positions_and_colors,
    probe_patch_channels_for_square_color,
    ranked_board_positions_with_probabilities_from_logits,
    resolve_square_intervention_type,
    sort_move_labels,
)
from train_tpr_probe import (  # noqa: E402
    load_saved_tpr_probe,
)


SUPPORTED_INTERVENED_SQUARE_COUNTS = (1, 2, 3, 4)
SQUARE_COUNT_WORDS = {
    1: "single",
    2: "two",
    3: "three",
    4: "four",
}
SQUARE_WEIGHT_LABELS = {
    2: "pair",
    3: "triplet",
    4: "quadruplet",
}
FIXED_PROBE_DIR = Path("probes/tpr")
FIXED_N_HEAD = 8
FIXED_BINDING_CONSTRUCTION_METHOD = "outer_products"
FIXED_PATCH_TARGET_NAME = "residual"
FIXED_PATCH_TARGET = "hook_resid_post"
FIXED_RESIDUAL_PROJECTION = "pinv"
FIXED_PREDICTION_MODE = "probability_threshold"
FIXED_PROBE_FILENAME_TEMPLATE = "resid_{layer}_tpr_r16_f8.pth"
DEFAULT_PATCH_LAYERS = (2, 3, 4, 5, 6, 7)
DEFAULT_SCALE_VALUES = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
)
FIXED_SQUARE_WEIGHT_VALUES = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
)
FIXED_PREDICTION_PROBABILITY_THRESHOLD = 1e-2
TOPK_EXTRA_LOGGED_MOVES = 5


def resolve_tpr_probe_path(*, layer: int) -> Path:
    resolved_probe_dir = FIXED_PROBE_DIR
    filename = FIXED_PROBE_FILENAME_TEMPLATE.format(layer=layer)
    direct_path = resolved_probe_dir / filename
    if direct_path.exists():
        return direct_path

    matches = sorted(resolved_probe_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {resolved_probe_dir}")
    if len(matches) > 1:
        joined_matches = ", ".join(str(match) for match in matches)
        raise FileNotFoundError(
            f"Found multiple matches for {filename} under {resolved_probe_dir}: "
            f"{joined_matches}"
        )
    return matches[0]


def parse_explicit_probe_pairs(
    raw_pairs: tuple[str, ...] | list[str]
) -> dict[int, Path]:
    pairs: dict[int, Path] = {}
    for raw_pair in raw_pairs:
        if "=" in raw_pair:
            layer_str, path_str = raw_pair.split("=", 1)
        elif ":" in raw_pair:
            layer_str, path_str = raw_pair.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --probe-pair {raw_pair!r}. Expected PATCH_LAYER=PATH."
            )

        patch_layer = int(layer_str.strip())
        probe_path = Path(path_str.strip()).expanduser()
        if patch_layer in pairs:
            raise ValueError(
                f"Duplicate --probe-pair specified for patch layer {patch_layer}"
            )
        if not probe_path.is_file():
            raise FileNotFoundError(f"TPR probe checkpoint not found: {probe_path}")
        pairs[patch_layer] = probe_path
    return pairs


def canonicalize_square_weight_tuple(
    square_weights: Sequence[float],
    *,
    expected_length: int,
) -> tuple[float, ...]:
    if len(square_weights) != expected_length:
        raise ValueError(
            f"Expected exactly {expected_length} square weights, got {len(square_weights)}"
        )

    normalized_weights = tuple(float(weight) for weight in square_weights)
    if any(not math.isfinite(weight) for weight in normalized_weights):
        raise ValueError(f"Square weights must be finite, got {square_weights!r}")

    max_abs_weight = max(abs(weight) for weight in normalized_weights)
    if max_abs_weight <= 0.0:
        raise ValueError(
            "At least one square weight must be non-zero so the direction is defined"
        )
    return tuple(weight / max_abs_weight for weight in normalized_weights)


def resolve_square_weight_tuples(
    *,
    square_weight_values: Sequence[float],
    num_intervened_squares: int,
) -> list[tuple[float, ...]]:
    if num_intervened_squares == 1:
        return [(1.0,)]

    raw_square_weight_tuples = [
        tuple(float(weight) for weight in square_weight_tuple)
        for square_weight_tuple in itertools.product(
            square_weight_values,
            repeat=num_intervened_squares,
        )
    ]

    resolved = []
    seen_keys: set[tuple[float, ...]] = set()
    for raw_square_weight_tuple in raw_square_weight_tuples:
        canonical_square_weight_tuple = canonicalize_square_weight_tuple(
            raw_square_weight_tuple,
            expected_length=num_intervened_squares,
        )
        dedupe_key = tuple(
            round(weight, 12) for weight in canonical_square_weight_tuple
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        resolved.append(canonical_square_weight_tuple)
    return resolved


def default_num_samples(num_intervened_squares: int) -> int:
    return 128 if num_intervened_squares == 1 else 256


def default_output_path(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "tpr_intervention_results.json"
    return (
        f"tpr_{SQUARE_COUNT_WORDS[num_intervened_squares]}_square_"
        "intervention_results.json"
    )


def default_require_reasonable_post_state(num_intervened_squares: int) -> bool:
    return num_intervened_squares >= 3


def square_count_phrase(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "single-square"
    return f"{SQUARE_COUNT_WORDS[num_intervened_squares]}-square"


def square_weight_label(num_intervened_squares: int) -> str:
    return SQUARE_WEIGHT_LABELS.get(num_intervened_squares, "tuple")


def square_weight_plural_label(num_intervened_squares: int) -> str:
    label = square_weight_label(num_intervened_squares)
    if label.endswith("y"):
        return f"{label[:-1]}ies"
    return f"{label}s"


def build_square_weight_selection_description(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "min(error, scale)"
    weight_terms = ", ".join(
        f"square_weight_{idx}" for idx in range(1, num_intervened_squares + 1)
    )
    ones = ", ".join("1" for _idx in range(num_intervened_squares))
    return f"min(error, scale, square_weight_deviation_from_({ones}), {weight_terms})"


def filter_benchmark_samples(
    samples: Sequence[dict],
    *,
    require_valid_change: bool,
    require_matching_valid_count: bool,
    require_reasonable_post_state: bool,
) -> list[dict]:
    filtered = []
    for sample in samples:
        pre_valids, post_valids, is_reasonable = (
            compute_pre_and_post_valids_for_squares(
                completion=sample["completion"],
                pos_ints=sample["pos_ints"],
                ori_colors=sample["ori_colors"],
                intervention_type=sample.get("intervention_type", "flip"),
            )
        )
        if require_reasonable_post_state and not is_reasonable:
            continue
        if require_valid_change and pre_valids == post_valids:
            continue
        if require_matching_valid_count and len(pre_valids) != len(post_valids):
            continue
        filtered.append(sample)
    return filtered


def generate_benchmark_from_data(
    data_path: str | Path,
    *,
    num_samples: int,
    min_prefix_len: int,
    seed: int,
    intervention_type: str = "flip",
    num_intervened_squares: int = 1,
    require_valid_change: bool,
    require_matching_valid_count: bool,
    require_reasonable_post_state: bool,
) -> list[dict]:
    raw_games = load_pickle_sequences(data_path)
    candidate_games = [game for game in raw_games if len(game) >= min_prefix_len + 1]
    if not candidate_games:
        raise ValueError(
            f"No games with at least {min_prefix_len + 1} moves found in {data_path}"
        )

    rng = random.Random(seed)
    benchmark = []
    seen = set()
    attempts = 0
    max_attempts = num_samples * 200

    while len(benchmark) < num_samples and attempts < max_attempts:
        attempts += 1
        game = rng.choice(candidate_games)
        prefix_len = rng.randint(min_prefix_len, min(len(game) - 1, 59))
        completion = [int(move) for move in game[:prefix_len]]

        board_state = OthelloBoardState()
        for move in completion:
            board_state.umpire(move)

        occupied_positions = [
            idx
            for idx, value in enumerate(board_state.state.flatten())
            if value != EMPTY and idx not in STARTING_SQUARES
        ]
        if len(occupied_positions) < num_intervened_squares:
            continue

        chosen_positions = [
            int(position)
            for position in rng.sample(
                occupied_positions,
                num_intervened_squares,
            )
        ]
        chosen_colors = [
            int(board_state.state.flatten()[pos_int] + 1)
            for pos_int in chosen_positions
        ]
        pos_ints, ori_colors = normalize_square_positions_and_colors(
            chosen_positions,
            chosen_colors,
            num_intervened_squares=num_intervened_squares,
        )
        resolved_intervention_type = resolve_square_intervention_type(
            intervention_type,
            rng=rng,
        )
        pre_valids, post_valids, is_reasonable = (
            compute_pre_and_post_valids_for_squares(
                completion=completion,
                pos_ints=pos_ints,
                ori_colors=ori_colors,
                intervention_type=resolved_intervention_type,
            )
        )
        if require_reasonable_post_state and not is_reasonable:
            continue
        if require_valid_change and pre_valids == post_valids:
            continue
        if require_matching_valid_count and len(pre_valids) != len(post_valids):
            continue

        key = (
            tuple(completion),
            tuple(pos_ints),
            tuple(ori_colors),
            resolved_intervention_type,
        )
        if key in seen:
            continue
        seen.add(key)
        benchmark.append(
            {
                "completion": completion,
                "pos_ints": pos_ints,
                "ori_colors": ori_colors,
                "intervention_type": resolved_intervention_type,
            }
        )

    if len(benchmark) < num_samples:
        raise ValueError(
            f"Could only build {len(benchmark)} intervention samples from {data_path}"
        )
    return benchmark


def load_benchmark(
    benchmark_path: str | None,
    *,
    data_path: str | Path,
    num_samples: int,
    min_prefix_len: int,
    seed: int,
    intervention_type: str = "flip",
    num_intervened_squares: int = 1,
    require_valid_change: bool,
    require_matching_valid_count: bool,
    require_reasonable_post_state: bool,
) -> list[dict]:
    if benchmark_path:
        raw_samples = load_benchmark_artifact(benchmark_path)
        if isinstance(raw_samples, dict):
            if "samples" in raw_samples:
                raw_samples = raw_samples["samples"]
            elif "benchmark" in raw_samples:
                raw_samples = raw_samples["benchmark"]
        if not isinstance(raw_samples, Sequence) or isinstance(
            raw_samples, (str, bytes)
        ):
            raise ValueError(
                f"Unsupported benchmark artifact schema in {benchmark_path!r}; "
                "expected a sequence of samples"
            )

        normalized_samples = normalize_benchmark_samples(
            raw_samples,
            num_intervened_squares=num_intervened_squares,
        )
        samples = assign_intervention_types_to_samples(
            normalized_samples,
            requested_intervention_type=intervention_type,
            seed=seed,
        )
        return filter_benchmark_samples(
            samples,
            require_valid_change=require_valid_change,
            require_matching_valid_count=require_matching_valid_count,
            require_reasonable_post_state=require_reasonable_post_state,
        )

    return generate_benchmark_from_data(
        data_path=data_path,
        num_samples=num_samples,
        min_prefix_len=min_prefix_len,
        seed=seed,
        intervention_type=intervention_type,
        num_intervened_squares=num_intervened_squares,
        require_valid_change=require_valid_change,
        require_matching_valid_count=require_matching_valid_count,
        require_reasonable_post_state=require_reasonable_post_state,
    )


def load_tpr_factors(
    probe_path: str | Path,
    *,
    d_model: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    probe, layer, _artifact = load_saved_tpr_probe(
        probe_path=probe_path,
        d_model=d_model,
        device=device,
    )
    return (
        probe.binding_map.detach().to(device),
        probe.role_embeddings.detach().to(device),
        probe.filler_embeddings.detach().to(device),
        layer,
    )


def build_binding_space_constraints_for_squares(
    role_embeddings: torch.Tensor,
    filler_embeddings: torch.Tensor,
    *,
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    move_idx: int,
    intervention_type: str = "flip",
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_pos_ints, normalized_ori_colors = normalize_square_positions_and_colors(
        pos_ints,
        ori_colors,
        num_intervened_squares=len(pos_ints),
    )

    selected_roles = []
    selected_filler_deltas = []
    for pos_int, ori_color in zip(
        normalized_pos_ints,
        normalized_ori_colors,
        strict=True,
    ):
        row, col = divmod(pos_int, 8)
        source_channel, target_channel = probe_patch_channels_for_square_color(
            original_color=ori_color,
            move_idx=move_idx,
            intervention_type=intervention_type,
        )
        selected_roles.append(role_embeddings[row, col, :])
        selected_filler_deltas.append(
            filler_embeddings[target_channel] - filler_embeddings[source_channel]
        )
    return torch.stack(selected_roles, dim=0), torch.stack(
        selected_filler_deltas, dim=0
    )


def build_binding_delta_as_sum_of_outer_products(
    selected_roles: torch.Tensor,
    selected_filler_deltas: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ir,if->rf", selected_roles, selected_filler_deltas)


def apply_square_weights_to_selected_filler_deltas(
    selected_filler_deltas: torch.Tensor,
    square_weights: Sequence[float],
) -> torch.Tensor:
    if len(square_weights) != selected_filler_deltas.shape[0]:
        raise ValueError(
            "Square weight count must match the number of selected squares; "
            f"got {len(square_weights)} weights for "
            f"{selected_filler_deltas.shape[0]} squares"
        )
    weight_tensor = selected_filler_deltas.new_tensor(
        [float(weight) for weight in square_weights]
    ).unsqueeze(-1)
    return selected_filler_deltas * weight_tensor


def solve_residual_delta_for_binding_delta(
    binding_map: torch.Tensor,
    delta_binding: torch.Tensor,
) -> torch.Tensor:
    delta_binding_flat = delta_binding.reshape(-1)
    binding_map_flat = binding_map.reshape(binding_map.shape[0], -1)
    return torch.linalg.pinv(binding_map_flat.T) @ delta_binding_flat


def tpr_binding_space_patch_direction_for_squares(
    binding_map: torch.Tensor,
    role_embeddings: torch.Tensor,
    filler_embeddings: torch.Tensor,
    *,
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    move_idx: int,
    intervention_type: str = "flip",
    square_weights: Sequence[float],
) -> torch.Tensor:
    selected_roles, selected_filler_deltas = (
        build_binding_space_constraints_for_squares(
            role_embeddings,
            filler_embeddings,
            pos_ints=pos_ints,
            ori_colors=ori_colors,
            move_idx=move_idx,
            intervention_type=intervention_type,
        )
    )
    delta_binding = build_binding_delta_as_sum_of_outer_products(
        selected_roles,
        apply_square_weights_to_selected_filler_deltas(
            selected_filler_deltas,
            square_weights,
        ),
    )
    direction = solve_residual_delta_for_binding_delta(
        binding_map,
        delta_binding,
    )
    return direction / direction.norm().clamp_min(1e-12)


@dataclass
class PredictionSnapshot:
    selected_moves: list[int]
    probability_by_move: dict[int, float]
    topk_preds: list[str]
    topk_plus_extra_preds: list[str]
    eval_preds: list[str]


def build_prediction_snapshot(
    ranked_with_probs: list[tuple[int, float]],
    *,
    num_reference_moves: int,
    probability_threshold: float,
) -> PredictionSnapshot:
    probability_by_move = dict(ranked_with_probs)
    selected_moves = [
        move
        for move, probability in ranked_with_probs
        if probability > probability_threshold
    ]
    return PredictionSnapshot(
        selected_moves=selected_moves,
        probability_by_move=probability_by_move,
        topk_preds=format_topk_predictions_alphanumerically(
            ranked_with_probs,
            num_moves=num_reference_moves,
        ),
        topk_plus_extra_preds=format_topk_predictions_alphanumerically(
            ranked_with_probs,
            num_moves=num_reference_moves + TOPK_EXTRA_LOGGED_MOVES,
        ),
        eval_preds=[
            format_move_prediction_with_probability(move, probability_by_move[move])
            for move in selected_moves
        ],
    )


@dataclass
class BestInterventionCandidate:
    scale: float
    square_weights: tuple[float, ...]
    snapshot: PredictionSnapshot
    false_positives: list[int]
    false_negatives: list[int]
    error: int


def candidate_selection_key(
    *,
    error: int,
    scale: float,
    square_weights: Sequence[float],
) -> tuple[float, ...]:
    if len(square_weights) == 1:
        return (float(error), float(scale))
    return (
        float(error),
        float(scale),
        float(sum(abs(weight - 1.0) for weight in square_weights)),
        *(float(weight) for weight in square_weights),
    )


def resolve_tpr_resources_for_patch_layers(
    *,
    probe_pairs: tuple[str, ...],
    d_model: int,
    device: torch.device,
) -> tuple[
    tuple[int, ...],
    dict[int, int],
    dict[int, str],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    explicit_probe_pairs = parse_explicit_probe_pairs(probe_pairs)
    patch_layers = (
        tuple(sorted(explicit_probe_pairs))
        if explicit_probe_pairs
        else DEFAULT_PATCH_LAYERS
    )

    probe_source_layers: dict[int, int] = {}
    probe_paths_by_patch_layer: dict[int, str] = {}
    binding_maps_by_patch_layer: dict[int, torch.Tensor] = {}
    role_embeddings_by_patch_layer: dict[int, torch.Tensor] = {}
    filler_embeddings_by_patch_layer: dict[int, torch.Tensor] = {}

    for patch_layer in patch_layers:
        probe_path = (
            explicit_probe_pairs[patch_layer]
            if explicit_probe_pairs
            else resolve_tpr_probe_path(layer=patch_layer)
        )
        binding_map, role_embeddings, filler_embeddings, loaded_layer = (
            load_tpr_factors(
                probe_path,
                d_model=d_model,
                device=device,
            )
        )
        if not explicit_probe_pairs and loaded_layer != patch_layer:
            raise ValueError(
                f"Loaded TPR checkpoint layer {loaded_layer} from {probe_path}, "
                f"expected layer {patch_layer}"
            )
        probe_source_layers[patch_layer] = loaded_layer
        probe_paths_by_patch_layer[patch_layer] = str(probe_path)
        binding_maps_by_patch_layer[patch_layer] = binding_map
        role_embeddings_by_patch_layer[patch_layer] = role_embeddings
        filler_embeddings_by_patch_layer[patch_layer] = filler_embeddings

    return (
        patch_layers,
        probe_source_layers,
        probe_paths_by_patch_layer,
        binding_maps_by_patch_layer,
        role_embeddings_by_patch_layer,
        filler_embeddings_by_patch_layer,
    )


@dataclass
class TPRInterventionConfig:
    checkpoint: str = "ckpts/synthetic_model.pth"
    data_path: str = "test_data"
    probe_pairs: tuple[str, ...] = ()
    benchmark_path: str | None = None
    output_path: str = "tpr_intervention_results.json"
    device: str = "auto"
    num_samples: int = 128
    min_prefix_len: int = 20
    num_intervened_squares: int = 1
    scale_values: tuple[float, ...] = DEFAULT_SCALE_VALUES
    scale: float | None = None
    intervention_type: str = "random"
    seed: int = 44
    verbose_limit: int = 5


def add_weight_summary_fields(
    summary: dict,
    *,
    num_intervened_squares: int,
    square_weight_tuples: Sequence[Sequence[float]],
    best_square_weight_counts: Counter[tuple[float, ...]],
) -> None:
    if num_intervened_squares == 1:
        return

    generic_counts = {
        ",".join(str(float(weight)) for weight in square_weight_tuple): count
        for square_weight_tuple, count in sorted(best_square_weight_counts.items())
    }
    generic_tuples = [
        [float(weight) for weight in square_weight_tuple]
        for square_weight_tuple in square_weight_tuples
    ]

    summary["square_weight_values"] = [
        float(weight) for weight in FIXED_SQUARE_WEIGHT_VALUES
    ]
    summary["square_weight_tuples"] = generic_tuples
    summary["num_square_weight_tuples"] = len(square_weight_tuples)
    summary["best_square_weight_selection"] = build_square_weight_selection_description(
        num_intervened_squares
    )
    summary["best_square_weight_counts"] = generic_counts

    label = square_weight_label(num_intervened_squares)
    plural_label = square_weight_plural_label(num_intervened_squares)
    summary[f"square_weight_{plural_label}"] = generic_tuples
    summary[f"num_square_weight_{plural_label}"] = len(square_weight_tuples)
    summary[f"best_square_weight_{label}_selection"] = summary[
        "best_square_weight_selection"
    ]
    summary[f"best_square_weight_{label}_counts"] = generic_counts


def run_interventions(config: TPRInterventionConfig) -> dict:
    print("TPR intervention config:")
    print(json.dumps(asdict(config), indent=2))
    seed_all(config.seed)

    device = torch.device(
        config.device
        if config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(
        {
            "model_path": config.checkpoint,
            "device": device,
            "n_head": FIXED_N_HEAD,
        }
    )
    convert_to_hooked_model(model)

    (
        patch_layers,
        probe_source_layers,
        probe_paths_by_patch_layer,
        binding_maps_by_patch_layer,
        role_embeddings_by_patch_layer,
        filler_embeddings_by_patch_layer,
    ) = resolve_tpr_resources_for_patch_layers(
        probe_pairs=config.probe_pairs,
        d_model=model.config.n_embd,
        device=device,
    )
    print(f"Forcing pseudoinverse residual projection for patch layers: {patch_layers}")
    require_reasonable_post_state = default_require_reasonable_post_state(
        config.num_intervened_squares
    )

    benchmark = load_benchmark(
        config.benchmark_path,
        data_path=config.data_path,
        num_samples=config.num_samples,
        min_prefix_len=config.min_prefix_len,
        seed=config.seed,
        intervention_type=config.intervention_type,
        num_intervened_squares=config.num_intervened_squares,
        require_valid_change=True,
        require_matching_valid_count=False,
        require_reasonable_post_state=require_reasonable_post_state,
    )

    false_positives = []
    false_negatives = []
    false_positives_null = []
    false_negatives_null = []
    exact_match_count = 0
    sample_best_results = []
    examples = []
    intervention_type_counts: Counter[str] = Counter()
    best_scale_counts: Counter[float] = Counter()
    best_square_weight_counts: Counter[tuple[float, ...]] = Counter()
    skipped_unchanged_valids = 0
    skipped_unreasonable_post_states = 0

    scale_values = tuple(float(scale) for scale in config.scale_values)
    square_weight_tuples = resolve_square_weight_tuples(
        square_weight_values=FIXED_SQUARE_WEIGHT_VALUES,
        num_intervened_squares=config.num_intervened_squares,
    )

    for sample in tqdm(
        benchmark,
        desc=f"{square_count_phrase(config.num_intervened_squares)} tpr interventions",
    ):
        completion = sample["completion"]
        if len(completion) > model.get_block_size():
            continue

        pos_ints, ori_colors = normalize_square_positions_and_colors(
            sample["pos_ints"],
            sample["ori_colors"],
            num_intervened_squares=config.num_intervened_squares,
        )
        sample_intervention_type = str(sample.get("intervention_type", "flip"))
        pre_valids, post_valids, is_reasonable = (
            compute_pre_and_post_valids_for_squares(
                completion=completion,
                pos_ints=pos_ints,
                ori_colors=ori_colors,
                intervention_type=sample_intervention_type,
            )
        )
        if require_reasonable_post_state and not is_reasonable:
            skipped_unreasonable_post_states += 1
            continue
        if not post_valids:
            continue
        if pre_valids == post_valids:
            skipped_unchanged_valids += 1
            continue
        intervention_type_counts[sample_intervention_type] += 1

        input_ids = torch.tensor(
            [encode_game_as_model_tokens(completion)],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            orig_logits, _ = model(input_ids)

        orig_snapshot = build_prediction_snapshot(
            ranked_board_positions_with_probabilities_from_logits(orig_logits),
            num_reference_moves=len(pre_valids),
            probability_threshold=FIXED_PREDICTION_PROBABILITY_THRESHOLD,
        )
        fp_null, fn_null, _null_error = compute_prediction_error(
            orig_snapshot.selected_moves,
            post_valids,
        )

        best_candidate = None
        for square_weight_tuple in square_weight_tuples:
            directions_by_patch_layer = {
                layer: tpr_binding_space_patch_direction_for_squares(
                    binding_maps_by_patch_layer[layer],
                    role_embeddings_by_patch_layer[layer],
                    filler_embeddings_by_patch_layer[layer],
                    pos_ints=pos_ints,
                    ori_colors=ori_colors,
                    move_idx=len(completion),
                    intervention_type=sample_intervention_type,
                    square_weights=square_weight_tuple,
                )
                for layer in patch_layers
            }
            for scale in scale_values:
                overrides = {
                    f"blocks.{layer}.{FIXED_PATCH_TARGET}": make_additive_patch(
                        direction=directions_by_patch_layer[layer],
                        scale=scale,
                    )
                    for layer in patch_layers
                }

                with intervene(model, overrides):
                    with torch.inference_mode():
                        patched_logits, _ = model(input_ids)

                patched_snapshot = build_prediction_snapshot(
                    ranked_board_positions_with_probabilities_from_logits(
                        patched_logits
                    ),
                    num_reference_moves=len(post_valids),
                    probability_threshold=FIXED_PREDICTION_PROBABILITY_THRESHOLD,
                )
                fp, fn, error = compute_prediction_error(
                    patched_snapshot.selected_moves,
                    post_valids,
                )
                selection_key = candidate_selection_key(
                    error=error,
                    scale=scale,
                    square_weights=square_weight_tuple,
                )
                if best_candidate is None or selection_key < candidate_selection_key(
                    error=best_candidate.error,
                    scale=best_candidate.scale,
                    square_weights=best_candidate.square_weights,
                ):
                    best_candidate = BestInterventionCandidate(
                        scale=scale,
                        square_weights=tuple(square_weight_tuple),
                        snapshot=patched_snapshot,
                        false_positives=fp,
                        false_negatives=fn,
                        error=error,
                    )

        if best_candidate is None:
            continue

        patched_snapshot = best_candidate.snapshot
        fp = best_candidate.false_positives
        fn = best_candidate.false_negatives
        best_scale = best_candidate.scale
        best_square_weights = best_candidate.square_weights

        false_positives.append(len(fp))
        false_negatives.append(len(fn))
        false_positives_null.append(len(fp_null))
        false_negatives_null.append(len(fn_null))
        exact_match_count += int(not fp and not fn)
        best_scale_counts[best_scale] += 1
        best_square_weight_counts[best_square_weights] += 1

        position_labels = [board_pos_to_label(pos_int) for pos_int in pos_ints]
        from_color_labels = [color_code_to_label(ori_color) for ori_color in ori_colors]
        sample_result = {
            "completion_length": len(completion),
            "intervention_positions": position_labels,
            "intervention_type": sample_intervention_type,
            "intervention_from_colors": from_color_labels,
            "best_scale": best_scale,
            "best_error": best_candidate.error,
            "best_false_positive": len(fp),
            "best_false_negative": len(fn),
            "orig_preds": orig_snapshot.eval_preds,
            "patched_preds": patched_snapshot.eval_preds,
            "orig_topk_preds": orig_snapshot.topk_preds,
            "patched_topk_preds": patched_snapshot.topk_preds,
            "orig_topk_plus_extra_preds": orig_snapshot.topk_plus_extra_preds,
            "patched_topk_plus_extra_preds": patched_snapshot.topk_plus_extra_preds,
        }
        if config.num_intervened_squares > 1:
            sample_result["best_square_weights"] = [
                float(weight) for weight in best_square_weights
            ]
            sample_result[
                f"best_square_weight_{square_weight_label(config.num_intervened_squares)}"
            ] = [float(weight) for weight in best_square_weights]
        sample_best_results.append(sample_result)

        if len(examples) < config.verbose_limit:
            sorted_pre_valids = sort_move_labels(
                [board_pos_to_label(move) for move in pre_valids]
            )
            sorted_post_valids = sort_move_labels(
                [board_pos_to_label(move) for move in post_valids]
            )
            formatted_orig_example_preds = format_selected_predictions_alphanumerically(
                orig_snapshot.selected_moves,
                probability_by_move=orig_snapshot.probability_by_move,
            )
            formatted_patched_example_preds = (
                format_selected_predictions_alphanumerically(
                    patched_snapshot.selected_moves,
                    probability_by_move=patched_snapshot.probability_by_move,
                )
            )
            comparison_table = format_move_comparison_table(
                {
                    "orig_preds": formatted_orig_example_preds,
                    "pre_valids": sorted_pre_valids,
                    "patched_preds": formatted_patched_example_preds,
                    "post_valids": sorted_post_valids,
                }
            )
            example = {
                "completion_length": len(completion),
                "intervention_positions": position_labels,
                "intervention_type": sample_intervention_type,
                "intervention_from_colors": from_color_labels,
                "best_scale": best_scale,
                "best_error": best_candidate.error,
                "pre_valids": sorted_pre_valids,
                "post_valids": sorted_post_valids,
                "orig_preds": formatted_orig_example_preds,
                "patched_preds": formatted_patched_example_preds,
                "orig_topk_preds": orig_snapshot.topk_preds,
                "patched_topk_preds": patched_snapshot.topk_preds,
                "orig_topk_plus_extra_preds": orig_snapshot.topk_plus_extra_preds,
                "patched_topk_plus_extra_preds": patched_snapshot.topk_plus_extra_preds,
                "orig_eval_preds": orig_snapshot.eval_preds,
                "patched_eval_preds": patched_snapshot.eval_preds,
                "comparison_table": comparison_table,
            }
            if config.num_intervened_squares > 1:
                example["best_square_weights"] = [
                    float(weight) for weight in best_square_weights
                ]
                example[
                    f"best_square_weight_{square_weight_label(config.num_intervened_squares)}"
                ] = [float(weight) for weight in best_square_weights]
            examples.append(example)

    errors = [
        false_positives[idx] + false_negatives[idx]
        for idx in range(len(false_positives))
    ]
    null_errors = [
        false_positives_null[idx] + false_negatives_null[idx]
        for idx in range(len(false_positives_null))
    ]

    intervention_kind = (
        "filler_delta"
        if config.num_intervened_squares == 1
        else (
            f"{SQUARE_COUNT_WORDS[config.num_intervened_squares]}_square_binding_"
            f"{FIXED_BINDING_CONSTRUCTION_METHOD}"
        )
    )
    summary = {
        "probe_kind": "tpr",
        "intervention_kind": intervention_kind,
        "intervention_space": "binding",
        "num_intervened_squares": config.num_intervened_squares,
        "binding_construction_method": FIXED_BINDING_CONSTRUCTION_METHOD,
        "patch_target_name": FIXED_PATCH_TARGET_NAME,
        "patch_target": FIXED_PATCH_TARGET,
        "residual_projection": FIXED_RESIDUAL_PROJECTION,
        "num_samples": len(errors),
        "exact_match_count": exact_match_count,
        "exact_match_percentage": (
            100.0 * exact_match_count / len(errors) if errors else None
        ),
        "mean_error": float(np.mean(errors)) if errors else None,
        "mean_false_positive": (
            float(np.mean(false_positives)) if false_positives else None
        ),
        "mean_false_negative": (
            float(np.mean(false_negatives)) if false_negatives else None
        ),
        "mean_null_error": float(np.mean(null_errors)) if null_errors else None,
        "mean_null_false_positive": (
            float(np.mean(false_positives_null)) if false_positives_null else None
        ),
        "mean_null_false_negative": (
            float(np.mean(false_negatives_null)) if false_negatives_null else None
        ),
        "prediction_mode": FIXED_PREDICTION_MODE,
        "prediction_probability_threshold": FIXED_PREDICTION_PROBABILITY_THRESHOLD,
        "prediction_probability_threshold_percent": (
            100.0 * FIXED_PREDICTION_PROBABILITY_THRESHOLD
        ),
        "scale_values": [float(scale) for scale in scale_values],
        "num_scale_values": len(scale_values),
        "topk_extra_logged_moves": TOPK_EXTRA_LOGGED_MOVES,
        "best_scale_selection": build_square_weight_selection_description(
            config.num_intervened_squares
        ),
        "best_scale_counts": {
            str(scale): count for scale, count in sorted(best_scale_counts.items())
        },
        "patch_layers": list(patch_layers),
        "probe_source_layers": probe_source_layers,
        "probe_paths": probe_paths_by_patch_layer,
        "scale": config.scale,
        "requested_intervention_type": config.intervention_type,
        "resolved_intervention_type_counts": {
            intervention_name: intervention_type_counts[intervention_name]
            for intervention_name in ACTUAL_INTERVENTION_TYPE_CHOICES
            if intervention_type_counts[intervention_name]
        },
        "sample_best_results": sample_best_results,
        "skipped_unchanged_valids": skipped_unchanged_valids,
        "examples": examples,
    }
    if require_reasonable_post_state:
        summary["post_state_reasonableness_filter"] = (
            "Keep only post-intervention board states with at least one legal move "
            "and a single 8-neighbor-connected occupied component."
        )
        summary["skipped_unreasonable_post_states"] = skipped_unreasonable_post_states

    add_weight_summary_fields(
        summary,
        num_intervened_squares=config.num_intervened_squares,
        square_weight_tuples=square_weight_tuples,
        best_square_weight_counts=best_square_weight_counts,
    )

    if examples:
        print("\nExample comparisons:")
        for example_idx, example in enumerate(examples, start=1):
            joined_positions = ", ".join(example["intervention_positions"])
            joined_colors = ", ".join(example["intervention_from_colors"])
            print(
                f"\nExample {example_idx}: interventions at {joined_positions} "
                f"(from {joined_colors}, {example['intervention_type']}) after "
                f"{example['completion_length']} moves"
            )
            if config.num_intervened_squares > 1:
                print(
                    f"best_scale: {example['best_scale']} "
                    f"best_square_weights: {example['best_square_weights']} "
                    f"(error={example['best_error']})"
                )
            else:
                print(
                    f"best_scale: {example['best_scale']} "
                    f"(error={example['best_error']})"
                )
            print(example["comparison_table"])
            print(
                f"orig_topk_plus_{TOPK_EXTRA_LOGGED_MOVES}: "
                + ", ".join(example["orig_topk_plus_extra_preds"])
            )
            print(
                f"patched_topk_plus_{TOPK_EXTRA_LOGGED_MOVES}: "
                + ", ".join(example["patched_topk_plus_extra_preds"])
            )

    with Path(config.output_path).open("w", encoding="utf-8") as file_p:
        json.dump(summary, file_p, indent=2)

    printable_summary = dict(summary)
    printable_summary.pop("examples", None)
    printable_summary.pop("sample_best_results", None)
    print(json.dumps(printable_summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-intervened-squares",
        type=int,
        choices=SUPPORTED_INTERVENED_SQUARE_COUNTS,
        default=1,
        help="How many board squares to intervene on per sample.",
    )
    parser.add_argument("--checkpoint", default=TPRInterventionConfig.checkpoint)
    parser.add_argument(
        "--data-path",
        default=TPRInterventionConfig.data_path,
        help="Path to a pickle dataset file or directory of pickle shards.",
    )
    parser.add_argument(
        "--probe-pair",
        action="append",
        default=None,
        metavar="PATCH_LAYER=PATH",
        help=(
            "Explicit mapping from a patched model layer to a TPR checkpoint path. "
            "Repeat this flag to specify multiple layer/path pairs."
        ),
    )
    parser.add_argument("--benchmark-path")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Defaults depend on --num-intervened-squares.",
    )
    parser.add_argument("--device", default=TPRInterventionConfig.device)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Defaults depend on --num-intervened-squares.",
    )
    parser.add_argument(
        "--min-prefix-len",
        type=int,
        default=TPRInterventionConfig.min_prefix_len,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Shortcut to evaluate one shared scale instead of the default sweep "
            "for the selected square count."
        ),
    )
    parser.add_argument(
        "--intervention-type",
        choices=INTERVENTION_TYPE_CHOICES,
        default=TPRInterventionConfig.intervention_type,
        help=(
            "Which board edit to emulate at the selected squares: `flip`, `empty`, "
            "or `random` to choose between them independently for each sample "
            "using --seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=TPRInterventionConfig.seed)
    parser.add_argument(
        "--verbose-limit",
        type=int,
        default=TPRInterventionConfig.verbose_limit,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    raw_probe_pairs = tuple(args.probe_pair or ())

    config = TPRInterventionConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        probe_pairs=raw_probe_pairs,
        benchmark_path=args.benchmark_path,
        output_path=args.output_path
        or default_output_path(args.num_intervened_squares),
        device=args.device,
        num_samples=args.num_samples
        or default_num_samples(args.num_intervened_squares),
        min_prefix_len=args.min_prefix_len,
        num_intervened_squares=args.num_intervened_squares,
        scale_values=(
            (float(args.scale),) if args.scale is not None else DEFAULT_SCALE_VALUES
        ),
        scale=args.scale,
        intervention_type=args.intervention_type,
        seed=args.seed,
        verbose_limit=args.verbose_limit,
    )
    run_interventions(config)


if __name__ == "__main__":
    main()
