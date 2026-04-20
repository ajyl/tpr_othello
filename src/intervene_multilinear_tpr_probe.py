"""Run multilinear TPR interventions on Othello GPT."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "hook_utils"))

from load_model import load_model  # noqa: E402
from hook_utils.intervene import intervene  # noqa: E402
from hook_utils.record_utils import convert_to_hooked_model  # noqa: E402
from hook_utils.util_funcs import seed_all  # noqa: E402
from intervene_probe import (  # noqa: E402
    ACTUAL_INTERVENTION_TYPE_CHOICES,
    board_pos_to_label,
    color_code_to_label,
    compute_pre_and_post_valids_for_squares,
    compute_prediction_error,
    encode_game_as_model_tokens,
    format_move_comparison_table,
    format_move_prediction_with_probability,
    format_selected_predictions_alphanumerically,
    format_topk_predictions_alphanumerically,
    make_additive_patch,
    normalize_square_positions_and_colors,
    ranked_board_positions_with_probabilities_from_logits,
    sort_move_labels,
)
from intervene_tpr_probe import (  # noqa: E402
    DEFAULT_SCALE_VALUES,
    SQUARE_COUNT_WORDS,
    TOPK_EXTRA_LOGGED_MOVES,
    build_square_weight_selection_description,
    candidate_selection_key,
    default_num_samples,
    load_benchmark,
    resolve_square_weight_tuples,
    square_count_phrase,
    square_weight_label,
    square_weight_plural_label,
)
from intervene_multilinear_tpr_probe_utils import (  # noqa: E402
    build_multilinear_binding_delta_as_sum_of_outer_products,
    build_multilinear_binding_space_constraints,
    resolve_multilinear_resources_for_patch_layers,
    shard_benchmark_samples,
    solve_residual_delta_for_binding_delta,
)


SUPPORTED_INTERVENED_SQUARE_COUNTS = (1, 2, 3, 4)
FIXED_PATCH_LAYERS = (2, 3, 4, 5, 6, 7)
FIXED_PROBE_DIR = "probes/tpr_multilinear"
FIXED_N_HEAD = 8
FIXED_ROW_DIM = 8
FIXED_COL_DIM = 8
FIXED_COLOR_DIM = 4
FIXED_USE_BIAS = False
FIXED_EXCLUDE_CENTER_SQUARES = False
FIXED_PREDICTION_PROBABILITY_THRESHOLD = 1e-2
FIXED_SQUARE_WEIGHT_VALUES = DEFAULT_SCALE_VALUES
FIXED_INTERVENTION_TYPE = "random"
FIXED_BINDING_CONSTRUCTION_METHOD = "outer_products"
FIXED_PATCH_TARGET_NAME = "residual"
FIXED_PATCH_TARGET = "hook_resid_post"
FIXED_RESIDUAL_PROJECTION = "pinv"
FIXED_PREDICTION_MODE = "probability_threshold"
FIXED_REQUIRE_MATCHING_VALID_COUNT = False


def default_output_path(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "multilinear_tpr_intervention_results.json"
    return (
        f"multilinear_tpr_{SQUARE_COUNT_WORDS[num_intervened_squares]}_square_"
        "intervention_results.json"
    )


def default_require_reasonable_post_state(num_intervened_squares: int) -> bool:
    return num_intervened_squares >= 3


def multilinear_patch_direction_for_squares(
    binding_map: torch.Tensor,
    row_embeddings: torch.Tensor,
    col_embeddings: torch.Tensor,
    color_embeddings: torch.Tensor,
    *,
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    move_idx: int,
    intervention_type: str,
    square_weights: Sequence[float],
) -> torch.Tensor:
    selected_spatial_features, selected_color_deltas = (
        build_multilinear_binding_space_constraints(
            row_embeddings=row_embeddings,
            col_embeddings=col_embeddings,
            color_embeddings=color_embeddings,
            pos_ints=pos_ints,
            ori_colors=ori_colors,
            move_idx=move_idx,
            intervention_type=intervention_type,
        )
    )
    weighted_color_deltas = selected_color_deltas.new_tensor(
        [float(weight) for weight in square_weights]
    ).unsqueeze(-1) * selected_color_deltas

    delta_binding = build_multilinear_binding_delta_as_sum_of_outer_products(
        selected_spatial_features=selected_spatial_features,
        selected_color_deltas=weighted_color_deltas,
        row_dim=row_embeddings.shape[-1],
        col_dim=col_embeddings.shape[-1],
    )
    direction = solve_residual_delta_for_binding_delta(
        binding_map=binding_map,
        delta_binding=delta_binding,
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


@dataclass
class MultilinearTPRInterventionConfig:
    checkpoint: str = "ckpts/synthetic_model.pth"
    data_path: str = "test_data"
    probe_pairs: tuple[str, ...] = ()
    benchmark_path: str | None = None
    output_path: str = "multilinear_tpr_intervention_results.json"
    device: str = "auto"
    probe_seed: int | None = None
    num_samples: int = 128
    min_prefix_len: int = 20
    num_intervened_squares: int = 1
    scale_values: tuple[float, ...] = DEFAULT_SCALE_VALUES
    scale: float | None = None
    intervention_type: str = FIXED_INTERVENTION_TYPE
    seed: int = 44
    verbose_limit: int = 5
    require_matching_valid_count: bool = FIXED_REQUIRE_MATCHING_VALID_COUNT
    require_reasonable_post_state: bool = False
    binding_construction_method: str = FIXED_BINDING_CONSTRUCTION_METHOD
    patch_target_name: str = FIXED_PATCH_TARGET_NAME
    patch_target: str = FIXED_PATCH_TARGET
    residual_projection: str = FIXED_RESIDUAL_PROJECTION
    prediction_mode: str = FIXED_PREDICTION_MODE
    num_benchmark_shards: int = 1
    benchmark_shard_index: int = 0


def add_weight_summary_fields(
    summary: dict,
    *,
    config: MultilinearTPRInterventionConfig,
    square_weight_tuples: Sequence[Sequence[float]],
    best_square_weight_counts: Counter[tuple[float, ...]],
) -> None:
    if config.num_intervened_squares == 1:
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
        config.num_intervened_squares
    )
    summary["best_square_weight_counts"] = generic_counts
    summary["square_weight_canonicalization"] = (
        "Per-square weights are canonicalized by dividing through by "
        "max(abs(w_i)) before evaluating candidates."
    )

    label = square_weight_label(config.num_intervened_squares)
    plural_label = square_weight_plural_label(config.num_intervened_squares)
    summary[f"square_weight_{plural_label}"] = generic_tuples
    summary[f"num_square_weight_{plural_label}"] = len(square_weight_tuples)
    summary[f"best_square_weight_{label}_selection"] = summary[
        "best_square_weight_selection"
    ]
    summary[f"best_square_weight_{label}_counts"] = generic_counts


def validate_config(config: MultilinearTPRInterventionConfig) -> None:
    if config.num_intervened_squares not in SUPPORTED_INTERVENED_SQUARE_COUNTS:
        raise ValueError(
            "--num-intervened-squares must be one of "
            f"{SUPPORTED_INTERVENED_SQUARE_COUNTS}"
        )
    for field_name, expected in (
        ("intervention_type", FIXED_INTERVENTION_TYPE),
        ("require_matching_valid_count", FIXED_REQUIRE_MATCHING_VALID_COUNT),
        ("binding_construction_method", FIXED_BINDING_CONSTRUCTION_METHOD),
        ("patch_target_name", FIXED_PATCH_TARGET_NAME),
        ("patch_target", FIXED_PATCH_TARGET),
        ("residual_projection", FIXED_RESIDUAL_PROJECTION),
        ("prediction_mode", FIXED_PREDICTION_MODE),
    ):
        actual = getattr(config, field_name)
        if actual != expected:
            raise ValueError(
                f"{field_name} is fixed to {expected!r} in this script; got {actual!r}"
            )


def run_interventions(config: MultilinearTPRInterventionConfig) -> dict:
    validate_config(config)
    print("Multilinear TPR intervention config:")
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
        probe_configs_by_patch_layer,
        binding_maps_by_patch_layer,
        row_embeddings_by_patch_layer,
        col_embeddings_by_patch_layer,
        color_embeddings_by_patch_layer,
    ) = resolve_multilinear_resources_for_patch_layers(
        probe_pairs=config.probe_pairs,
        probe_seed=config.probe_seed,
        patch_layers=FIXED_PATCH_LAYERS,
        probe_dir=FIXED_PROBE_DIR,
        row_dim=FIXED_ROW_DIM,
        col_dim=FIXED_COL_DIM,
        color_dim=FIXED_COLOR_DIM,
        use_bias=FIXED_USE_BIAS,
        exclude_center_squares=FIXED_EXCLUDE_CENTER_SQUARES,
        d_model=model.config.n_embd,
        device=device,
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
        require_matching_valid_count=config.require_matching_valid_count,
        require_reasonable_post_state=config.require_reasonable_post_state,
    )
    benchmark = shard_benchmark_samples(
        benchmark,
        num_benchmark_shards=config.num_benchmark_shards,
        benchmark_shard_index=config.benchmark_shard_index,
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
        desc=f"{square_count_phrase(config.num_intervened_squares)} multilinear tpr interventions",
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
        if config.require_reasonable_post_state and not is_reasonable:
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
                layer: multilinear_patch_direction_for_squares(
                    binding_map=binding_maps_by_patch_layer[layer],
                    row_embeddings=row_embeddings_by_patch_layer[layer],
                    col_embeddings=col_embeddings_by_patch_layer[layer],
                    color_embeddings=color_embeddings_by_patch_layer[layer],
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
                    f"blocks.{layer}.{config.patch_target}": make_additive_patch(
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
                "comparison_table": format_move_comparison_table(
                    {
                        "orig_preds": formatted_orig_example_preds,
                        "pre_valids": sorted_pre_valids,
                        "patched_preds": formatted_patched_example_preds,
                        "post_valids": sorted_post_valids,
                    }
                ),
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
        "color_delta"
        if config.num_intervened_squares == 1
        else (
            f"{SQUARE_COUNT_WORDS[config.num_intervened_squares]}_square_binding_"
            f"{FIXED_BINDING_CONSTRUCTION_METHOD}"
        )
    )
    summary = {
        "probe_kind": "multilinear_tensor_product",
        "intervention_kind": intervention_kind,
        "intervention_space": "binding",
        "num_intervened_squares": config.num_intervened_squares,
        "binding_construction_method": config.binding_construction_method,
        "patch_target_name": config.patch_target_name,
        "patch_target": config.patch_target,
        "residual_projection": config.residual_projection,
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
        "prediction_mode": config.prediction_mode,
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
        "probe_dir": FIXED_PROBE_DIR,
        "probe_seed": config.probe_seed,
        "probe_source_layers": probe_source_layers,
        "probe_paths": probe_paths_by_patch_layer,
        "binding_to_residual_sources": {
            layer: "pseudoinverse_forced" for layer in patch_layers
        },
        "probe_configs": probe_configs_by_patch_layer,
        "row_dim": FIXED_ROW_DIM,
        "col_dim": FIXED_COL_DIM,
        "color_dim": FIXED_COLOR_DIM,
        "use_bias": FIXED_USE_BIAS,
        "exclude_center_squares": FIXED_EXCLUDE_CENTER_SQUARES,
        "scale": config.scale,
        "requested_intervention_type": config.intervention_type,
        "resolved_intervention_type_counts": {
            intervention_name: intervention_type_counts[intervention_name]
            for intervention_name in ACTUAL_INTERVENTION_TYPE_CHOICES
            if intervention_type_counts[intervention_name]
        },
        "sample_best_results": sample_best_results,
        "skipped_unchanged_valids": skipped_unchanged_valids,
        "skipped_mismatched_valid_counts": 0,
        "examples": examples,
        "num_benchmark_shards": config.num_benchmark_shards,
        "benchmark_shard_index": config.benchmark_shard_index,
    }
    if config.require_reasonable_post_state:
        summary["post_state_reasonableness_filter"] = (
            "Keep only post-intervention board states with at least one legal move "
            "and a single 8-neighbor-connected occupied component."
        )
        summary["skipped_unreasonable_post_states"] = skipped_unreasonable_post_states

    add_weight_summary_fields(
        summary,
        config=config,
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
    parser.add_argument("--checkpoint", default=MultilinearTPRInterventionConfig.checkpoint)
    parser.add_argument(
        "--data-path",
        default=MultilinearTPRInterventionConfig.data_path,
        help="Path to a pickle dataset file or directory of pickle shards.",
    )
    parser.add_argument(
        "--probe-pair",
        action="append",
        default=None,
        metavar="PATCH_LAYER=PATH",
        help=(
            "Explicit mapping from a patched model layer to a multilinear TPR "
            "checkpoint path. Repeat this flag to specify multiple layer/path pairs."
        ),
    )
    parser.add_argument("--benchmark-path")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Defaults depend on --num-intervened-squares.",
    )
    parser.add_argument("--device", default=MultilinearTPRInterventionConfig.device)
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=MultilinearTPRInterventionConfig.probe_seed,
        help="Optional seed suffix used when resolving multilinear checkpoints.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Defaults depend on --num-intervened-squares.",
    )
    parser.add_argument(
        "--min-prefix-len",
        type=int,
        default=MultilinearTPRInterventionConfig.min_prefix_len,
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
    parser.add_argument("--seed", type=int, default=MultilinearTPRInterventionConfig.seed)
    parser.add_argument(
        "--verbose-limit",
        type=int,
        default=MultilinearTPRInterventionConfig.verbose_limit,
    )
    parser.add_argument(
        "--num-benchmark-shards",
        type=int,
        default=MultilinearTPRInterventionConfig.num_benchmark_shards,
        help="Shard the benchmark deterministically by sample index.",
    )
    parser.add_argument(
        "--benchmark-shard-index",
        type=int,
        default=MultilinearTPRInterventionConfig.benchmark_shard_index,
        help="Which benchmark shard to run when --num-benchmark-shards > 1.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    raw_probe_pairs = tuple(args.probe_pair or ())

    num_intervened_squares = args.num_intervened_squares
    config = MultilinearTPRInterventionConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        probe_pairs=raw_probe_pairs,
        benchmark_path=args.benchmark_path,
        output_path=args.output_path or default_output_path(num_intervened_squares),
        device=args.device,
        probe_seed=args.probe_seed,
        num_samples=args.num_samples or default_num_samples(num_intervened_squares),
        min_prefix_len=args.min_prefix_len,
        num_intervened_squares=num_intervened_squares,
        scale_values=(
            (float(args.scale),) if args.scale is not None else DEFAULT_SCALE_VALUES
        ),
        scale=args.scale,
        intervention_type=FIXED_INTERVENTION_TYPE,
        seed=args.seed,
        verbose_limit=args.verbose_limit,
        require_matching_valid_count=FIXED_REQUIRE_MATCHING_VALID_COUNT,
        require_reasonable_post_state=default_require_reasonable_post_state(
            num_intervened_squares
        ),
        binding_construction_method=FIXED_BINDING_CONSTRUCTION_METHOD,
        patch_target_name=FIXED_PATCH_TARGET_NAME,
        patch_target=FIXED_PATCH_TARGET,
        residual_projection=FIXED_RESIDUAL_PROJECTION,
        prediction_mode=FIXED_PREDICTION_MODE,
        num_benchmark_shards=args.num_benchmark_shards,
        benchmark_shard_index=args.benchmark_shard_index,
    )
    run_interventions(config)


if __name__ == "__main__":
    main()
