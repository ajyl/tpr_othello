"""Run configurable 1-4 square linear-probe interventions on Othello GPT."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import itertools
import json
import pickle
import random
import re
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


ROW_LABELS = "ABCDEFGH"
COLUMN_LABELS = tuple(str(idx) for idx in range(1, 9))
STARTING_SQUARES = (27, 28, 35, 36)
BLACK = 1
WHITE = -1
EMPTY = 0
ACTUAL_INTERVENTION_TYPE_CHOICES = ("flip", "empty")
INTERVENTION_TYPE_CHOICES = ACTUAL_INTERVENTION_TYPE_CHOICES + ("random",)
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
DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
DEFAULT_PROBE_DIR = Path("probes/linear")
DEFAULT_PATCH_LAYERS = (2, 3, 4, 5, 6, 7)
FIXED_PATCH_TARGET_NAME = "residual"
FIXED_PATCH_TARGET = "hook_resid_post"
FIXED_PREDICTION_MODE = "probability_threshold"
DEFAULT_PREDICTION_PROBABILITY_THRESHOLD = 1e-2
TOPK_EXTRA_LOGGED_MOVES = 5
SUPPORTED_INTERVENED_SQUARE_COUNTS = (1, 2, 3, 4)
SQUARE_COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
}
SCALE_SUMMARY_LABELS = {
    1: "scale",
    2: "pair",
    3: "triplet",
    4: "quadruplet",
}
COLOR_CODE_TO_LABEL = {
    0: "white",
    2: "black",
}
COLOR_LABEL_TO_CODE = {
    "white": 0,
    "w": 0,
    "black": 2,
    "b": 2,
}


def build_itos() -> dict[int, int]:
    itos = {0: -100}
    for idx in range(1, 28):
        itos[idx] = idx - 1
    for idx in range(28, 34):
        itos[idx] = idx + 1
    for idx in range(34, 61):
        itos[idx] = idx + 3
    return itos


def build_stoi() -> dict[int, int]:
    itos = build_itos()
    stoi = {board_pos: token_id for token_id, board_pos in itos.items()}
    stoi[-1] = 0
    return stoi


ITOS = build_itos()
STOI = build_stoi()


class OthelloBoardState:
    def __init__(self):
        board = np.zeros((8, 8), dtype=np.int8)
        board[3, 4] = BLACK
        board[3, 3] = WHITE
        board[4, 3] = BLACK
        board[4, 4] = WHITE
        self.state = board
        self.next_hand_color = BLACK

    def copy(self) -> "OthelloBoardState":
        other = OthelloBoardState()
        other.state = np.copy(self.state)
        other.next_hand_color = self.next_hand_color
        return other

    def get_valid_moves(self) -> list[int]:
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            move_type = self.tentative_move(move)
            if move_type == 1:
                regular_moves.append(move)
            elif move_type == 2:
                forfeit_moves.append(move)
        return regular_moves if regular_moves else forfeit_moves

    def tentative_move(self, move: int) -> int:
        row, col = divmod(move, 8)
        if self.state[row, col] != EMPTY:
            return 0

        color = self.next_hand_color
        to_flip = self._collect_flips(row, col, color)
        if to_flip:
            return 1

        to_flip = self._collect_flips(row, col, -color)
        return 2 if to_flip else 0

    def umpire(self, move: int) -> None:
        row, col = divmod(move, 8)
        if self.state[row, col] != EMPTY:
            raise ValueError(f"Illegal move {move}: square already occupied")

        color = self.next_hand_color
        to_flip = self._collect_flips(row, col, color)
        if not to_flip:
            color = -color
            self.next_hand_color = -self.next_hand_color
            to_flip = self._collect_flips(row, col, color)

        if not to_flip:
            if not self.get_valid_moves():
                raise ValueError("Game should have ended before this move")
            raise ValueError(f"Illegal move {move}: no discs flipped")

        for flip_row, flip_col in to_flip:
            self.state[flip_row, flip_col] *= -1
        self.state[row, col] = color
        self.next_hand_color *= -1

    def _collect_flips(self, row: int, col: int, color: int) -> list[tuple[int, int]]:
        flips = []
        for d_row, d_col in DIRECTIONS:
            buffer = []
            cur_row, cur_col = row, col
            while True:
                cur_row += d_row
                cur_col += d_col
                if not (0 <= cur_row < 8 and 0 <= cur_col < 8):
                    break
                square = self.state[cur_row, cur_col]
                if square == EMPTY:
                    break
                if square == color:
                    flips.extend(buffer)
                    break
                buffer.append((cur_row, cur_col))
        return flips


def encode_game_as_model_tokens(raw_moves: Sequence[int]) -> list[int]:
    return [STOI[int(move)] for move in raw_moves]


def board_pos_to_label(idx: int) -> str:
    return f"{ROW_LABELS[idx // 8]}{COLUMN_LABELS[idx % 8]}".lower()


def board_label_to_pos(label: str) -> int:
    label = label.strip().upper()
    return ROW_LABELS.index(label[0]) * 8 + (int(label[1]) - 1)


def sort_move_labels(labels: Sequence[str]) -> list[str]:
    def key(label: str) -> tuple[int, int]:
        label = label.strip().upper()
        return ROW_LABELS.index(label[0]), int(label[1:])

    return sorted(labels, key=key)


def format_move_comparison_table(columns: dict[str, Sequence[str]]) -> str:
    headers = list(columns.keys())
    rows = max((len(values) for values in columns.values()), default=0)
    widths = {
        header: max(len(header), *(len(value) for value in columns[header]), 0)
        for header in headers
    }

    header_line = " | ".join(f"{header:<{widths[header]}}" for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = []
    for row_idx in range(rows):
        body.append(
            " | ".join(
                f"{(columns[header][row_idx] if row_idx < len(columns[header]) else ''):<{widths[header]}}"
                for header in headers
            )
        )
    return "\n".join([header_line, separator, *body])


def resolve_square_intervention_type(
    requested_intervention_type: str,
    *,
    rng: random.Random | None = None,
) -> str:
    if requested_intervention_type in ACTUAL_INTERVENTION_TYPE_CHOICES:
        return requested_intervention_type
    if requested_intervention_type == "random":
        if rng is None:
            raise ValueError("An RNG is required when intervention_type='random'")
        return rng.choice(ACTUAL_INTERVENTION_TYPE_CHOICES)
    raise ValueError(
        "Unsupported intervention_type "
        f"{requested_intervention_type!r}; expected one of {INTERVENTION_TYPE_CHOICES}"
    )


def assign_intervention_types_to_samples(
    samples: Sequence[dict],
    *,
    requested_intervention_type: str,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    resolved_samples = []
    for sample in samples:
        resolved_sample = dict(sample)
        resolved_sample["intervention_type"] = resolve_square_intervention_type(
            requested_intervention_type,
            rng=rng,
        )
        resolved_samples.append(resolved_sample)
    return resolved_samples


def board_value_after_intervention(
    ori_color: int,
    intervention_type: str,
) -> int:
    if ori_color not in (0, 2):
        raise ValueError(f"Expected square color encoded as 0 or 2, got {ori_color}")
    if intervention_type == "flip":
        return int(2 - ori_color) - 1
    if intervention_type == "empty":
        return EMPTY
    raise ValueError(
        f"Unsupported intervention_type {intervention_type!r}; expected one of "
        f"{ACTUAL_INTERVENTION_TYPE_CHOICES}"
    )


def apply_intervention_to_board_state(
    board_state: "OthelloBoardState",
    *,
    pos_int: int,
    ori_color: int,
    intervention_type: str,
) -> None:
    row, col = divmod(int(pos_int), 8)
    board_state.state[row, col] = board_value_after_intervention(
        ori_color=int(ori_color),
        intervention_type=intervention_type,
    )


def apply_interventions_to_board_state(
    board_state: "OthelloBoardState",
    *,
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    intervention_type: str,
) -> None:
    for pos_int, ori_color in zip(pos_ints, ori_colors, strict=True):
        apply_intervention_to_board_state(
            board_state,
            pos_int=int(pos_int),
            ori_color=int(ori_color),
            intervention_type=intervention_type,
        )


def probe_patch_channels_for_square_color(
    original_color: int,
    move_idx: int,
    intervention_type: str = "flip",
) -> tuple[int, int]:
    if original_color not in (0, 2):
        raise ValueError(f"Expected square color encoded as 0 or 2, got {original_color}")

    if move_idx % 2 == 0:
        source_channel = 1 if original_color == 0 else 2
    else:
        source_channel = 2 if original_color == 0 else 1

    if intervention_type == "flip":
        target_channel = 3 - source_channel
    elif intervention_type == "empty":
        target_channel = 0
    else:
        raise ValueError(
            f"Unsupported intervention_type {intervention_type!r}; expected one of "
            f"{ACTUAL_INTERVENTION_TYPE_CHOICES}"
        )
    return source_channel, target_channel


def color_code_to_label(code: int) -> str:
    if int(code) not in COLOR_CODE_TO_LABEL:
        raise ValueError(f"Unsupported square color code: {code}")
    return COLOR_CODE_TO_LABEL[int(code)]


def normalize_square_color_code(raw_color: object) -> int:
    if isinstance(raw_color, str):
        normalized = raw_color.strip().lower()
        if normalized in COLOR_LABEL_TO_CODE:
            return COLOR_LABEL_TO_CODE[normalized]
        raw_color = int(normalized)

    color_code = int(raw_color)
    if color_code in (0, 2):
        return color_code
    if color_code in (WHITE, BLACK):
        return color_code + 1
    raise ValueError(
        "Expected square color encoded as 0/2 or black/white labels, "
        f"got {raw_color!r}"
    )


def ensure_sequence_length(
    raw_value: object,
    *,
    expected_length: int,
    value_name: str,
) -> list[object]:
    if isinstance(raw_value, np.ndarray):
        values = raw_value.tolist()
    elif isinstance(raw_value, torch.Tensor):
        values = raw_value.tolist()
    elif isinstance(raw_value, str) and expected_length > 1 and "," in raw_value:
        values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        values = list(raw_value)
    else:
        values = [raw_value]

    if len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} {value_name} values, got {len(values)}: {raw_value!r}"
        )
    return values


def normalize_square_positions_and_colors(
    raw_positions: object,
    raw_colors: object,
    *,
    num_intervened_squares: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    raw_position_values = ensure_sequence_length(
        raw_positions,
        expected_length=num_intervened_squares,
        value_name="intervention position",
    )
    raw_color_values = ensure_sequence_length(
        raw_colors,
        expected_length=num_intervened_squares,
        value_name="intervention color",
    )

    pos_ints = []
    for raw_position in raw_position_values:
        if isinstance(raw_position, str):
            pos_ints.append(board_label_to_pos(raw_position))
        else:
            pos_ints.append(int(raw_position))

    ori_colors = [normalize_square_color_code(raw_color) for raw_color in raw_color_values]

    if len(set(pos_ints)) != len(pos_ints):
        raise ValueError(f"Intervention positions must be distinct, got {pos_ints}")

    ordered_pairs = sorted(zip(pos_ints, ori_colors, strict=True), key=lambda pair: pair[0])
    return (
        tuple(int(pos_int) for pos_int, _ori_color in ordered_pairs),
        tuple(int(ori_color) for _pos_int, ori_color in ordered_pairs),
    )


def pick_first_present_key(sample: dict, keys: Sequence[str]) -> object:
    for key in keys:
        if key in sample:
            return sample[key]
    raise KeyError(f"Expected one of keys {list(keys)} in benchmark sample: {sample.keys()}")


def normalize_benchmark_samples(
    raw_samples: Sequence[dict],
    *,
    num_intervened_squares: int,
) -> list[dict]:
    normalized = []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            raise ValueError(
                f"Unsupported benchmark sample type {type(sample)}; expected dict entries"
            )

        completion = [
            int(move)
            for move in pick_first_present_key(sample, ("completion", "history"))
        ]
        raw_positions = pick_first_present_key(
            sample,
            (
                "pos_ints",
                "pos_int",
                "intervention_positions",
                "intervention_position",
            ),
        )
        raw_colors = pick_first_present_key(
            sample,
            (
                "ori_colors",
                "ori_color",
                "intervention_from_colors",
                "intervention_from",
            ),
        )
        pos_ints, ori_colors = normalize_square_positions_and_colors(
            raw_positions,
            raw_colors,
            num_intervened_squares=num_intervened_squares,
        )

        normalized_sample = {
            "completion": completion,
            "pos_ints": pos_ints,
            "ori_colors": ori_colors,
        }
        if "intervention_type" in sample:
            normalized_sample["intervention_type"] = str(sample["intervention_type"])
        normalized.append(normalized_sample)
    return normalized


def compute_pre_and_post_valids_for_squares(
    completion: Sequence[int],
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    intervention_type: str = "flip",
) -> tuple[set[int], set[int], bool]:
    board_state = OthelloBoardState()
    for move in completion:
        board_state.umpire(int(move))

    pre_valids = set(board_state.get_valid_moves())
    modified_board = board_state.copy()
    apply_interventions_to_board_state(
        modified_board,
        pos_ints=pos_ints,
        ori_colors=ori_colors,
        intervention_type=intervention_type,
    )
    post_valids = set(modified_board.get_valid_moves())
    return (
        pre_valids,
        post_valids,
        is_reasonable_post_intervention_state(modified_board, post_valids),
    )


def is_reasonable_post_intervention_state(
    board_state: "OthelloBoardState",
    post_valids: set[int],
) -> bool:
    if not post_valids:
        return False
    return has_single_occupied_component(board_state.state)


def has_single_occupied_component(board: np.ndarray) -> bool:
    occupied = {
        (row, col)
        for row in range(8)
        for col in range(8)
        if int(board[row, col]) != EMPTY
    }
    if not occupied:
        return False

    to_visit = {next(iter(occupied))}
    visited = set()
    while to_visit:
        row, col = to_visit.pop()
        if (row, col) in visited:
            continue
        visited.add((row, col))
        for d_row, d_col in DIRECTIONS:
            neighbor = (row + d_row, col + d_col)
            if neighbor in occupied and neighbor not in visited:
                to_visit.add(neighbor)
    return visited == occupied


def filter_benchmark_samples(
    samples: Sequence[dict],
) -> list[dict]:
    filtered = []
    for sample in samples:
        pre_valids, post_valids, is_reasonable = compute_pre_and_post_valids_for_squares(
            completion=sample["completion"],
            pos_ints=sample["pos_ints"],
            ori_colors=sample["ori_colors"],
            intervention_type=sample.get("intervention_type", "flip"),
        )
        if not is_reasonable or pre_valids == post_valids:
            continue
        filtered.append(sample)
    return filtered


def generate_benchmark_from_data(
    data_path: str | Path,
    num_samples: int,
    min_prefix_len: int,
    seed: int,
    intervention_type: str = "flip",
    num_intervened_squares: int = 1,
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

        chosen_positions = rng.sample(occupied_positions, num_intervened_squares)
        chosen_colors = [
            int(board_state.state.flatten()[pos_int] + 1) for pos_int in chosen_positions
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
        pre_valids, post_valids, is_reasonable = compute_pre_and_post_valids_for_squares(
            completion=completion,
            pos_ints=pos_ints,
            ori_colors=ori_colors,
            intervention_type=resolved_intervention_type,
        )
        if not is_reasonable or pre_valids == post_valids:
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


def load_benchmark_artifact(path: str | Path) -> object:
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {resolved_path}")
    if resolved_path.suffix.lower() == ".json":
        with resolved_path.open("r", encoding="utf-8") as file_p:
            return json.load(file_p)
    with resolved_path.open("rb") as file_p:
        return pickle.load(file_p)


def load_benchmark(
    benchmark_path: str | None,
    data_path: str | Path,
    num_samples: int,
    min_prefix_len: int,
    seed: int,
    intervention_type: str = "flip",
    num_intervened_squares: int = 1,
) -> list[dict]:
    if benchmark_path:
        raw_samples = load_benchmark_artifact(benchmark_path)
        if isinstance(raw_samples, dict):
            if "samples" in raw_samples:
                raw_samples = raw_samples["samples"]
            elif "benchmark" in raw_samples:
                raw_samples = raw_samples["benchmark"]
        if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
            raise ValueError(
                f"Unsupported benchmark artifact schema in {benchmark_path!r}; expected a sequence of samples"
            )

        samples = assign_intervention_types_to_samples(
            normalize_benchmark_samples(
                raw_samples,
                num_intervened_squares=num_intervened_squares,
            ),
            requested_intervention_type=intervention_type,
            seed=seed,
        )
        return filter_benchmark_samples(samples)

    return generate_benchmark_from_data(
        data_path=data_path,
        num_samples=num_samples,
        min_prefix_len=min_prefix_len,
        seed=seed,
        intervention_type=intervention_type,
        num_intervened_squares=num_intervened_squares,
    )


LINEAR_PROBE_FILENAME_RE = re.compile(r"resid_(\d+)_linear\.pth$")


def parse_explicit_probe_pairs(raw_pairs: tuple[str, ...] | list[str]) -> dict[int, Path]:
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
            raise FileNotFoundError(f"Linear probe checkpoint not found: {probe_path}")
        pairs[patch_layer] = probe_path
    return pairs


def load_linear_probe_from_path(
    probe_path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    resolved_path = Path(probe_path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Linear probe checkpoint not found: {resolved_path}")

    artifact = torch.load(resolved_path, map_location=device)
    if isinstance(artifact, dict):
        if "probe" not in artifact:
            raise KeyError(f"Missing probe tensor in {resolved_path}")
        probe = artifact["probe"]
        if "layer" in artifact:
            return probe.to(device), int(artifact["layer"])
    else:
        probe = artifact

    match = LINEAR_PROBE_FILENAME_RE.fullmatch(resolved_path.name)
    if match is None:
        raise ValueError(
            f"Could not infer probe layer from checkpoint path {resolved_path}. "
            "Expected an artifact with a 'layer' field or a filename like "
            "resid_<layer>_linear.pth."
        )
    return probe.to(device), int(match.group(1))


def resolve_linear_probe_resources_for_patch_layers(
    *,
    probe_pairs: tuple[str, ...],
    device: torch.device,
) -> tuple[list[int], dict[int, int], dict[int, str], dict[int, torch.Tensor]]:
    explicit_probe_pairs = parse_explicit_probe_pairs(probe_pairs)
    probe_source_layers: dict[int, int] = {}
    probe_paths_by_patch_layer: dict[int, str] = {}
    probes_by_patch_layer: dict[int, torch.Tensor] = {}

    if explicit_probe_pairs:
        patch_layers = sorted(explicit_probe_pairs)
        for patch_layer in patch_layers:
            probe_path = explicit_probe_pairs[patch_layer]
            probe, loaded_layer = load_linear_probe_from_path(
                probe_path=probe_path,
                device=device,
            )
            probe_source_layers[patch_layer] = loaded_layer
            probe_paths_by_patch_layer[patch_layer] = str(probe_path)
            probes_by_patch_layer[patch_layer] = probe
        return (
            patch_layers,
            probe_source_layers,
            probe_paths_by_patch_layer,
            probes_by_patch_layer,
        )

    patch_layers = list(DEFAULT_PATCH_LAYERS)
    for layer in patch_layers:
        probe_path = DEFAULT_PROBE_DIR / f"resid_{layer}_linear.pth"
        probe, loaded_layer = load_linear_probe_from_path(
            probe_path=probe_path,
            device=device,
        )
        if loaded_layer != layer:
            raise ValueError(
                f"Loaded linear probe layer {loaded_layer} from {probe_path}, "
                f"expected layer {layer}"
            )
        probe_source_layers[layer] = layer
        probe_paths_by_patch_layer[layer] = str(probe_path)
        probes_by_patch_layer[layer] = probe
    return (
        patch_layers,
        probe_source_layers,
        probe_paths_by_patch_layer,
        probes_by_patch_layer,
    )


def normalized_patch_directions_for_squares(
    probe: torch.Tensor,
    *,
    pos_ints: Sequence[int],
    ori_colors: Sequence[int],
    move_idx: int,
    intervention_type: str = "flip",
) -> tuple[torch.Tensor, ...]:
    directions = []
    for pos_int, ori_color in zip(pos_ints, ori_colors, strict=True):
        row, col = divmod(int(pos_int), 8)
        _source_channel, target_channel = probe_patch_channels_for_square_color(
            original_color=int(ori_color),
            move_idx=move_idx,
            intervention_type=intervention_type,
        )
        direction = probe[0, :, row, col, target_channel]
        directions.append(direction / direction.norm().clamp_min(1e-12))
    return tuple(directions)


def combine_scaled_square_directions(
    square_directions: Sequence[torch.Tensor],
    scale_combination: Sequence[float],
) -> torch.Tensor:
    if len(square_directions) != len(scale_combination):
        raise ValueError(
            "Mismatched square direction and scale lengths: "
            f"{len(square_directions)} != {len(scale_combination)}"
        )

    combined_direction = None
    for scale, direction in zip(scale_combination, square_directions, strict=True):
        contribution = float(scale) * direction
        combined_direction = (
            contribution if combined_direction is None else combined_direction + contribution
        )
    if combined_direction is None:
        raise ValueError("At least one square direction is required")
    return combined_direction


def ranked_board_positions_with_probabilities_from_logits(
    logits: torch.Tensor,
) -> list[tuple[int, float]]:
    final_logits = logits[0, -1]
    token_logits = final_logits[1:]
    token_probs = torch.softmax(token_logits, dim=0)
    ranked_token_ids = torch.argsort(token_logits, descending=True) + 1
    return [
        (int(ITOS[int(token_id)]), float(token_probs[int(token_id) - 1]))
        for token_id in ranked_token_ids.tolist()
    ]


def format_move_prediction_with_probability(move: int, probability: float) -> str:
    return f"{board_pos_to_label(move)} ({probability:.4f})"


def select_threshold_predictions(
    ranked_with_probs: list[tuple[int, float]],
    *,
    probability_threshold: float,
) -> list[int]:
    return [
        move
        for move, probability in ranked_with_probs
        if probability > probability_threshold
    ]


def format_topk_predictions_alphanumerically(
    ranked_with_probs: list[tuple[int, float]],
    *,
    num_moves: int,
) -> list[str]:
    topk_preds = sorted(ranked_with_probs[:num_moves], key=lambda pair: pair[0])
    return [
        format_move_prediction_with_probability(move, probability)
        for move, probability in topk_preds
    ]


def format_selected_predictions_alphanumerically(
    selected_moves: Sequence[int],
    *,
    probability_by_move: dict[int, float],
) -> list[str]:
    return [
        format_move_prediction_with_probability(move, probability_by_move[move])
        for move in sorted(selected_moves)
    ]


@dataclass
class PredictionSnapshot:
    selected_moves: list[int]
    probability_by_move: dict[int, float]
    topk_preds: list[str]
    topk_plus_extra_preds: list[str]
    eval_preds: list[str]


@dataclass
class BestInterventionCandidate:
    scale_combination: tuple[float, ...]
    snapshot: PredictionSnapshot
    false_positives: list[int]
    false_negatives: list[int]
    error: int


def build_prediction_snapshot(
    ranked_with_probs: list[tuple[int, float]],
    *,
    num_reference_moves: int,
    probability_threshold: float,
) -> PredictionSnapshot:
    probability_by_move = dict(ranked_with_probs)
    selected_moves = select_threshold_predictions(
        ranked_with_probs,
        probability_threshold=probability_threshold,
    )
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
        eval_preds=format_selected_predictions_alphanumerically(
            selected_moves,
            probability_by_move=probability_by_move,
        ),
    )


def compute_prediction_error(
    predicted_moves: Sequence[int],
    target_moves: set[int],
) -> tuple[list[int], list[int], int]:
    false_positives = [move for move in predicted_moves if move not in target_moves]
    false_negatives = [move for move in target_moves if move not in predicted_moves]
    return (
        false_positives,
        false_negatives,
        len(false_positives) + len(false_negatives),
    )


def find_best_intervention_candidate(
    *,
    model,
    input_ids: torch.Tensor,
    patch_layers: Sequence[int],
    square_directions_by_patch_layer: dict[int, tuple[torch.Tensor, ...]],
    post_valids: set[int],
    scale_combinations: Sequence[tuple[float, ...]],
    probability_threshold: float,
) -> BestInterventionCandidate | None:
    best_candidate = None
    for scale_combination in scale_combinations:
        overrides = {
            f"blocks.{layer}.{FIXED_PATCH_TARGET}": make_additive_patch(
                direction=combine_scaled_square_directions(
                    square_directions_by_patch_layer[layer],
                    scale_combination,
                ),
                scale=1.0,
            )
            for layer in patch_layers
        }

        with intervene(model, overrides):
            with torch.inference_mode():
                patched_logits, _ = model(input_ids)

        snapshot = build_prediction_snapshot(
            ranked_board_positions_with_probabilities_from_logits(patched_logits),
            num_reference_moves=len(post_valids),
            probability_threshold=probability_threshold,
        )
        false_positives, false_negatives, error = compute_prediction_error(
            snapshot.selected_moves,
            post_valids,
        )
        selection_key = (error, sum(scale_combination), *scale_combination)
        if best_candidate is None or selection_key < (
            best_candidate.error,
            sum(best_candidate.scale_combination),
            *best_candidate.scale_combination,
        ):
            best_candidate = BestInterventionCandidate(
                scale_combination=scale_combination,
                snapshot=snapshot,
                false_positives=false_positives,
                false_negatives=false_negatives,
                error=error,
            )
    return best_candidate


def make_additive_patch(direction: torch.Tensor, scale: float):
    def patch(orig_output, _inputs, _module, _name):
        if not isinstance(orig_output, torch.Tensor):
            raise TypeError(f"Expected tensor output, got {type(orig_output)}")
        output = orig_output.clone()
        output[:, -1, :] = output[:, -1, :] + scale * direction.to(
            device=output.device,
            dtype=output.dtype,
        )
        return output

    return patch


def default_num_samples(num_intervened_squares: int) -> int:
    return 128 if num_intervened_squares == 1 else 256


def default_output_path(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "intervention_results.json"
    return (
        f"linear_{SQUARE_COUNT_WORDS[num_intervened_squares]}_square_"
        "intervention_results.json"
    )


def square_count_phrase(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "single-square"
    return f"{SQUARE_COUNT_WORDS[num_intervened_squares]}-square"


def build_scale_selection_description(num_intervened_squares: int) -> str:
    if num_intervened_squares == 1:
        return "min(error, scale)"
    scale_terms = ", ".join(
        f"scale_{idx}" for idx in range(1, num_intervened_squares + 1)
    )
    return f"min(error, total_scale, {scale_terms})"


def format_scale_combination(scale_combination: Sequence[float]) -> str:
    if len(scale_combination) == 1:
        return str(float(scale_combination[0]))
    return ",".join(str(float(scale)) for scale in scale_combination)


def best_scale_summary_key(num_intervened_squares: int, *, counts: bool) -> str:
    if num_intervened_squares == 1:
        return "best_scale_counts" if counts else "best_scale_selection"
    label = SCALE_SUMMARY_LABELS[num_intervened_squares]
    suffix = "counts" if counts else "selection"
    return f"best_scale_{label}_{suffix}"


@dataclass
class InterventionConfig:
    checkpoint: str = "ckpts/synthetic_model.pth"
    data_path: str = "test_data"
    probe_pairs: tuple[str, ...] = ()
    benchmark_path: str | None = None
    output_path: str = "intervention_results.json"
    device: str = "auto"
    num_samples: int = 1000
    min_prefix_len: int = 20
    num_intervened_squares: int = 1
    scale_values: tuple[float, ...] = DEFAULT_SCALE_VALUES
    scale: float | None = None
    intervention_type: str = "random"
    seed: int = 44
    verbose_limit: int = 5
    prediction_probability_threshold: float = DEFAULT_PREDICTION_PROBABILITY_THRESHOLD


def run_interventions(config: InterventionConfig) -> dict:
    print("Intervention config:")
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
            "n_head": 8,
        }
    )
    convert_to_hooked_model(model)

    (
        patch_layers,
        probe_source_layers,
        probe_paths_by_patch_layer,
        probes_by_patch_layer,
    ) = resolve_linear_probe_resources_for_patch_layers(
        probe_pairs=config.probe_pairs,
        device=device,
    )
    benchmark = load_benchmark(
        benchmark_path=config.benchmark_path,
        data_path=config.data_path,
        num_samples=config.num_samples,
        min_prefix_len=config.min_prefix_len,
        seed=config.seed,
        intervention_type=config.intervention_type,
        num_intervened_squares=config.num_intervened_squares,
    )

    false_positives = []
    false_negatives = []
    false_positives_null = []
    false_negatives_null = []
    exact_match_count = 0
    null_exact_match_count = 0
    sample_best_results = []
    examples = []
    intervention_type_counts: Counter[str] = Counter()
    best_scale_combination_counts: Counter[tuple[float, ...]] = Counter()
    skipped_unchanged_valids = 0
    skipped_unreasonable_post_states = 0

    scale_values = tuple(float(scale) for scale in config.scale_values)
    scale_combinations = [
        tuple(float(scale) for scale in scale_combination)
        for scale_combination in itertools.product(
            scale_values,
            repeat=config.num_intervened_squares,
        )
    ]

    for sample in tqdm(
        benchmark,
        desc=f"{square_count_phrase(config.num_intervened_squares)} linear interventions",
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
        pre_valids, post_valids, is_reasonable = compute_pre_and_post_valids_for_squares(
            completion=completion,
            pos_ints=pos_ints,
            ori_colors=ori_colors,
            intervention_type=sample_intervention_type,
        )
        if not is_reasonable:
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

        square_directions_by_patch_layer = {
            layer: normalized_patch_directions_for_squares(
                probe=probes_by_patch_layer[layer],
                pos_ints=pos_ints,
                ori_colors=ori_colors,
                move_idx=len(completion),
                intervention_type=sample_intervention_type,
            )
            for layer in patch_layers
        }

        orig_snapshot = build_prediction_snapshot(
            ranked_board_positions_with_probabilities_from_logits(orig_logits),
            num_reference_moves=len(pre_valids),
            probability_threshold=config.prediction_probability_threshold,
        )
        fp_null, fn_null, _null_error = compute_prediction_error(
            orig_snapshot.selected_moves,
            post_valids,
        )

        best_candidate = find_best_intervention_candidate(
            model=model,
            input_ids=input_ids,
            patch_layers=patch_layers,
            square_directions_by_patch_layer=square_directions_by_patch_layer,
            post_valids=post_valids,
            scale_combinations=scale_combinations,
            probability_threshold=config.prediction_probability_threshold,
        )

        if best_candidate is None:
            continue

        patched_snapshot = best_candidate.snapshot
        fp = best_candidate.false_positives
        fn = best_candidate.false_negatives
        best_scale_combination = best_candidate.scale_combination

        false_positives.append(len(fp))
        false_negatives.append(len(fn))
        false_positives_null.append(len(fp_null))
        false_negatives_null.append(len(fn_null))
        exact_match_count += int(not fp and not fn)
        null_exact_match_count += int(not fp_null and not fn_null)
        best_scale_combination_counts[best_scale_combination] += 1

        position_labels = [board_pos_to_label(pos_int) for pos_int in pos_ints]
        from_color_labels = [color_code_to_label(ori_color) for ori_color in ori_colors]
        sample_best_results.append(
            {
                "completion_length": len(completion),
                "intervention_positions": position_labels,
                "intervention_type": sample_intervention_type,
                "intervention_from_colors": from_color_labels,
                "best_scales": list(best_scale_combination),
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
        )

        if len(examples) < config.verbose_limit:
            sorted_pre_valids = sort_move_labels(
                [board_pos_to_label(move) for move in pre_valids]
            )
            sorted_post_valids = sort_move_labels(
                [board_pos_to_label(move) for move in post_valids]
            )
            comparison_table = format_move_comparison_table(
                {
                    "orig_preds": orig_snapshot.eval_preds,
                    "pre_valids": sorted_pre_valids,
                    "patched_preds": patched_snapshot.eval_preds,
                    "post_valids": sorted_post_valids,
                }
            )
            examples.append(
                {
                    "completion_length": len(completion),
                    "intervention_positions": position_labels,
                    "intervention_type": sample_intervention_type,
                    "intervention_from_colors": from_color_labels,
                    "best_scales": list(best_scale_combination),
                    "best_error": best_candidate.error,
                    "pre_valids": sorted_pre_valids,
                    "post_valids": sorted_post_valids,
                    "orig_preds": orig_snapshot.eval_preds,
                    "patched_preds": patched_snapshot.eval_preds,
                    "orig_topk_preds": orig_snapshot.topk_preds,
                    "patched_topk_preds": patched_snapshot.topk_preds,
                    "orig_topk_plus_extra_preds": orig_snapshot.topk_plus_extra_preds,
                    "patched_topk_plus_extra_preds": patched_snapshot.topk_plus_extra_preds,
                    "orig_eval_preds": orig_snapshot.eval_preds,
                    "patched_eval_preds": patched_snapshot.eval_preds,
                    "comparison_table": comparison_table,
                }
            )

    errors = [
        false_positives[idx] + false_negatives[idx]
        for idx in range(len(false_positives))
    ]
    null_errors = [
        false_positives_null[idx] + false_negatives_null[idx]
        for idx in range(len(false_positives_null))
    ]

    scale_summary = {
        format_scale_combination(scale_combination): count
        for scale_combination, count in sorted(best_scale_combination_counts.items())
    }
    scale_selection_description = build_scale_selection_description(
        config.num_intervened_squares
    )

    summary = {
        "probe_kind": "linear",
        "intervention_kind": (
            f"{SQUARE_COUNT_WORDS[config.num_intervened_squares]}_square_linear_target_sum"
        ),
        "num_intervened_squares": config.num_intervened_squares,
        "intervention_space": FIXED_PATCH_TARGET_NAME,
        "patch_target_name": FIXED_PATCH_TARGET_NAME,
        "patch_target": FIXED_PATCH_TARGET,
        "num_samples": len(errors),
        "exact_match_count": exact_match_count,
        "exact_match_percentage": (
            100.0 * exact_match_count / len(errors) if errors else None
        ),
        "null_exact_match_count": null_exact_match_count,
        "null_exact_match_percentage": (
            100.0 * null_exact_match_count / len(errors) if errors else None
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
        "prediction_probability_threshold": config.prediction_probability_threshold,
        "prediction_probability_threshold_percent": (
            100.0 * config.prediction_probability_threshold
        ),
        "scale_values": list(scale_values),
        "num_scale_combinations": len(scale_combinations),
        "topk_extra_logged_moves": TOPK_EXTRA_LOGGED_MOVES,
        "best_scale_combination_selection": scale_selection_description,
        "best_scale_combination_counts": scale_summary,
        "patch_layers": patch_layers,
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
        "skipped_unreasonable_post_states": skipped_unreasonable_post_states,
        "post_state_reasonableness_filter": (
            "Keep only post-intervention board states with at least one legal move "
            "and a single 8-neighbor-connected occupied component."
        ),
        "examples": examples,
    }
    summary[best_scale_summary_key(config.num_intervened_squares, counts=False)] = (
        scale_selection_description
    )
    summary[best_scale_summary_key(config.num_intervened_squares, counts=True)] = (
        scale_summary
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
    parser.add_argument("--checkpoint", default=InterventionConfig.checkpoint)
    parser.add_argument(
        "--data-path",
        default=InterventionConfig.data_path,
        help="Path to a pickle dataset file or directory of pickle shards.",
    )
    parser.add_argument(
        "--probe-pair",
        action="append",
        default=None,
        metavar="PATCH_LAYER=PATH",
        help=(
            "Explicit mapping from a patched model layer to a linear probe checkpoint "
            "path. Repeat this flag to specify multiple layer/path pairs."
        ),
    )
    parser.add_argument(
        "--benchmark-path",
        help="Optional pickle/json benchmark file. Otherwise samples are generated from --data-path.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the JSON summary. Defaults depend on --num-intervened-squares.",
    )
    parser.add_argument("--device", default=InterventionConfig.device)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of benchmark samples to evaluate. Defaults depend on --num-intervened-squares.",
    )
    parser.add_argument(
        "--min-prefix-len",
        type=int,
        default=InterventionConfig.min_prefix_len,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Legacy shortcut to evaluate one shared per-square scale instead of the "
            f"default cartesian sweep over {list(DEFAULT_SCALE_VALUES)}."
        ),
    )
    parser.add_argument(
        "--intervention-type",
        choices=INTERVENTION_TYPE_CHOICES,
        default=InterventionConfig.intervention_type,
        help=(
            "Which board edit to emulate at the selected squares: `flip`, `empty`, "
            "or `random` to choose between them independently for each sample "
            "using --seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=InterventionConfig.seed)
    parser.add_argument(
        "--verbose-limit",
        type=int,
        default=InterventionConfig.verbose_limit,
    )
    parser.add_argument(
        "--prediction-probability-threshold",
        type=float,
        default=DEFAULT_PREDICTION_PROBABILITY_THRESHOLD,
        help="Probability threshold for predicted move selection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.prediction_probability_threshold <= 1.0:
        raise ValueError(
            "--prediction-probability-threshold must be between 0 and 1"
        )

    raw_probe_pairs = tuple(args.probe_pair or ())
    num_intervened_squares = int(args.num_intervened_squares)

    config = InterventionConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        probe_pairs=raw_probe_pairs,
        benchmark_path=args.benchmark_path,
        output_path=args.output_path or default_output_path(num_intervened_squares),
        device=args.device,
        num_samples=(
            int(args.num_samples)
            if args.num_samples is not None
            else default_num_samples(num_intervened_squares)
        ),
        min_prefix_len=args.min_prefix_len,
        num_intervened_squares=num_intervened_squares,
        scale_values=(
            (float(args.scale),)
            if args.scale is not None
            else InterventionConfig.scale_values
        ),
        scale=args.scale,
        intervention_type=args.intervention_type,
        seed=args.seed,
        verbose_limit=args.verbose_limit,
        prediction_probability_threshold=args.prediction_probability_threshold,
    )
    run_interventions(config)


if __name__ == "__main__":
    main()
