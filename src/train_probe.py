"""Train linear board-state probes on Othello GPT activations using hook_utils."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_model import load_model  # noqa: E402
from data_utils import load_pickle_sequences  # noqa: E402
from hook_utils import (
    convert_to_hooked_model,
    record_activations,
    seed_all,
)  # noqa: E402


STARTING_SQUARES = (27, 28, 35, 36)
BOARD_ROWS = 8
BOARD_COLS = 8
BOARD_LABEL_OPTIONS = 3
ROW_LABELS = "ABCDEFGH"
BLACK = 1
WHITE = -1
EMPTY = 0
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


def build_itos() -> dict[int, int]:
    """Map model token ids back to board positions, skipping the four starting squares."""
    itos = {0: -100}
    for idx in range(1, 28):
        itos[idx] = idx - 1
    for idx in range(28, 34):
        itos[idx] = idx + 1
    for idx in range(34, 61):
        itos[idx] = idx + 3
    return itos


def build_stoi() -> dict[int, int]:
    """Map board positions to model token ids."""
    itos = build_itos()
    stoi = {board_pos: token_id for token_id, board_pos in itos.items()}
    stoi[-1] = 0
    return stoi


STOI = build_stoi()


class OthelloBoardState:
    """Minimal board engine used to construct probe labels from move sequences."""

    def __init__(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        board[3, 4] = BLACK
        board[3, 3] = WHITE
        board[4, 3] = BLACK
        board[4, 4] = WHITE
        self.state = board
        self.next_hand_color = BLACK

    def get_valid_moves(self) -> list[int]:
        regular_moves = []
        forfeit_moves = []
        for move in range(BOARD_ROWS * BOARD_COLS):
            move_type = self.tentative_move(move)
            if move_type == 1:
                regular_moves.append(move)
            elif move_type == 2:
                forfeit_moves.append(move)
        return regular_moves if regular_moves else forfeit_moves

    def tentative_move(self, move: int) -> int:
        row, col = divmod(move, BOARD_COLS)
        if self.state[row, col] != EMPTY:
            return 0

        color = self.next_hand_color
        to_flip = self._collect_flips(row, col, color)
        if to_flip:
            return 1

        to_flip = self._collect_flips(row, col, -color)
        return 2 if to_flip else 0

    def umpire(self, move: int) -> None:
        row, col = divmod(move, BOARD_COLS)
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
                if not (0 <= cur_row < BOARD_ROWS and 0 <= cur_col < BOARD_COLS):
                    break
                square = self.state[cur_row, cur_col]
                if square == EMPTY:
                    break
                if square == color:
                    flips.extend(buffer)
                    break
                buffer.append((cur_row, cur_col))
        return flips


def seq_to_state_stack(raw_moves: Sequence[int]) -> np.ndarray:
    board = OthelloBoardState()
    states = []
    for move in raw_moves:
        board.umpire(int(move))
        states.append(np.copy(board.state))
    return np.stack(states, axis=0)


def build_state_stack(raw_games: Sequence[Sequence[int]]) -> Tensor:
    """Construct board states after each move for a batch of games."""
    stacks = [seq_to_state_stack(game) for game in raw_games]
    return torch.tensor(np.stack(stacks), dtype=torch.long)


def state_stack_to_one_hot_threeway(state_stack: Tensor) -> Tensor:
    """Convert board states into the three-way labeling used by the reference probe."""
    one_hot = torch.zeros(
        1,
        state_stack.shape[0],
        state_stack.shape[1],
        BOARD_ROWS,
        BOARD_COLS,
        BOARD_LABEL_OPTIONS,
        device=state_stack.device,
        dtype=torch.float32,
    )
    one_hot[..., 0] = state_stack == EMPTY
    one_hot[0, :, 0::2, ..., 1] = (state_stack == BLACK)[:, 0::2]
    one_hot[0, :, 1::2, ..., 1] = (state_stack == WHITE)[:, 1::2]
    one_hot[0, :, 0::2, ..., 2] = (state_stack == WHITE)[:, 0::2]
    one_hot[0, :, 1::2, ..., 2] = (state_stack == BLACK)[:, 1::2]
    return one_hot


def encode_game_as_model_tokens(raw_moves: Sequence[int]) -> list[int]:
    return [STOI[int(move)] for move in raw_moves]


def macro_f1_score(preds: Tensor, targets: Tensor, num_classes: int) -> float:
    preds = preds.view(-1)
    targets = targets.view(-1)
    f1_scores = []
    for class_id in range(num_classes):
        pred_mask = preds == class_id
        target_mask = targets == class_id
        tp = (pred_mask & target_mask).sum().item()
        fp = (pred_mask & ~target_mask).sum().item()
        fn = (~pred_mask & target_mask).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / num_classes)


def format_square_accuracy_board(square_accuracy: Tensor) -> str:
    """Render per-square accuracies as an 8x8 board."""
    square_accuracy = square_accuracy.detach().cpu()
    header = "     " + "  ".join(f"{col:>5}" for col in range(1, BOARD_COLS + 1))
    rows = [header]
    has_nan = False
    for row_idx in range(BOARD_ROWS):
        rendered_values = []
        for value in square_accuracy[row_idx]:
            value_item = float(value.item())
            if math.isnan(value_item):
                rendered_values.append("  N/A")
                has_nan = True
            else:
                rendered_values.append(f"{100 * value_item:5.1f}")
        values = "  ".join(rendered_values)
        rows.append(f"{ROW_LABELS[row_idx]}  {values}")
    if has_nan:
        rows.append("     values are accuracy percentages; N/A indicates excluded squares")
    else:
        rows.append("     values are accuracy percentages")
    return "\n".join(rows)


@dataclass
class ProbeConfig:
    checkpoint: str = "ckpts/synthetic_model.pth"
    data_path: str = "train_data"
    output_dir: str = "probes/linear"
    device: str = "auto"
    random_init: bool = False
    n_head: int = 8
    lr: float = 1e-2
    wd: float = 1e-2
    batch_size: int = 128
    valid_every: int = 100
    pos_start: int = 0
    pos_end: int | None = None
    num_epochs: int = 1
    valid_size: int = 512
    test_size: int = 1000
    valid_patience: int = 10
    seed: int = 1111
    max_games: int | None = None
    max_layers: int | None = None
    train_layers: str | None = "5,6,7"


def load_probe_dataset(
    data_path: str | Path,
    block_size: int,
    valid_size: int,
    test_size: int,
    seed: int,
    max_games: int | None = None,
) -> dict[str, list]:
    raw_games = load_pickle_sequences(data_path)

    expected_game_len = block_size + 1
    full_games = [game for game in raw_games if len(game) == expected_game_len]
    if max_games is not None:
        full_games = full_games[:max_games]

    if not full_games:
        raise ValueError(
            f"No full games of length {expected_game_len} found in {data_path}"
        )

    encoded_games = torch.tensor(
        [encode_game_as_model_tokens(game) for game in full_games],
        dtype=torch.long,
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(full_games), generator=generator).tolist()
    full_games = [full_games[idx] for idx in permutation]
    encoded_games = encoded_games[permutation]

    if valid_size + test_size >= len(full_games):
        raise ValueError(
            "valid_size + test_size must be smaller than the number of available games"
        )

    train_end = len(full_games) - valid_size - test_size
    valid_end = len(full_games) - test_size

    return {
        "train_raw": full_games[:train_end],
        "train_tokens": encoded_games[:train_end],
        "valid_raw": full_games[train_end:valid_end],
        "valid_tokens": encoded_games[train_end:valid_end],
        "test_raw": full_games[valid_end:],
        "test_tokens": encoded_games[valid_end:],
    }


def resolve_position_slice(
    pos_start: int,
    pos_end: int | None,
    block_size: int,
) -> tuple[int, int]:
    resolved_start = int(pos_start)
    resolved_end = block_size if pos_end is None else int(pos_end)

    if not (0 <= resolved_start < block_size):
        raise ValueError(
            f"pos_start must be in [0, {block_size - 1}], got {resolved_start}"
        )
    if not (1 <= resolved_end <= block_size):
        raise ValueError(
            f"pos_end must be in [1, {block_size}] or omitted, got {resolved_end}"
        )
    if resolved_start >= resolved_end:
        raise ValueError(
            f"Empty position slice: pos_start={resolved_start}, pos_end={resolved_end}, "
            f"block_size={block_size}"
        )
    return resolved_start, resolved_end


def select_layers(config: ProbeConfig, model) -> list[int]:
    if config.train_layers:
        layers = [
            int(piece) for piece in config.train_layers.split(",") if piece.strip()
        ]
    else:
        layers = list(range(model.config.n_layer))
    if config.max_layers is not None:
        layers = layers[: config.max_layers]
    return layers


def compute_probe_logits(residual: Tensor, probe: Tensor) -> Tensor:
    return torch.einsum(
        "bpd,mdrco->mbprco",
        residual,
        probe,
    )


def build_square_selection_mask(
    *,
    exclude_center_squares: bool,
    device: torch.device | str,
) -> Tensor:
    mask = torch.ones(BOARD_ROWS, BOARD_COLS, device=device, dtype=torch.bool)
    if exclude_center_squares:
        for board_pos in STARTING_SQUARES:
            row_idx, col_idx = divmod(board_pos, BOARD_COLS)
            mask[row_idx, col_idx] = False
    return mask


def compute_probe_loss(
    probe_logits: Tensor,
    state_stack_one_hot: Tensor,
    square_mask: Tensor | None = None,
) -> Tensor:
    log_probs = probe_logits.log_softmax(dim=-1)
    correct_log_probs = (log_probs * state_stack_one_hot).sum(dim=-1).mean(
        dim=1
    ) * BOARD_LABEL_OPTIONS
    if square_mask is not None:
        selected_correct_log_probs = correct_log_probs[0].reshape(
            correct_log_probs.shape[1], -1
        )[:, square_mask.reshape(-1)]
        return -selected_correct_log_probs.mean(dim=0).sum()
    return -correct_log_probs[0].mean(dim=0).sum()


def evaluate_probe(
    model,
    probe: Tensor,
    layer: int,
    games_tokens: Tensor,
    games_raw: Sequence[Sequence[int]],
    batch_size: int,
    pos_start: int,
    pos_end: int,
    device: torch.device,
    exclude_center_squares: bool = False,
) -> dict[str, float]:
    module_name = f"blocks.{layer}.hook_resid_post"
    total_weight = 0
    total_loss = 0.0
    total_accuracy = 0.0
    square_num_correct = torch.zeros(BOARD_ROWS, BOARD_COLS, dtype=torch.float64)
    square_num_total = torch.zeros(BOARD_ROWS, BOARD_COLS, dtype=torch.float64)
    all_preds = []
    all_targets = []
    square_mask = build_square_selection_mask(
        exclude_center_squares=exclude_center_squares,
        device=device,
    )
    square_mask_flat = square_mask.reshape(-1)
    selected_square_count = int(square_mask.sum().item())

    with torch.inference_mode():
        for batch_start in range(0, len(games_raw), batch_size):
            batch_end = min(batch_start + batch_size, len(games_raw))
            batch_raw = games_raw[batch_start:batch_end]
            batch_tokens = games_tokens[batch_start:batch_end].to(device)
            state_stack = build_state_stack(batch_raw)[:, pos_start:pos_end].to(device)
            state_stack_one_hot = state_stack_to_one_hot_threeway(state_stack)

            with record_activations(model, [module_name]) as cache:
                model(batch_tokens[:, :-1])
            resid_post = cache[module_name][0][:, pos_start:pos_end].to(device)
            probe_logits = compute_probe_logits(resid_post, probe)
            loss = compute_probe_loss(
                probe_logits,
                state_stack_one_hot,
                square_mask=square_mask,
            )

            preds = probe_logits.argmax(dim=-1)
            targets = state_stack_one_hot.argmax(dim=-1)
            correct = (preds == targets).float()
            flat_correct = correct.reshape(correct.shape[0], correct.shape[1], correct.shape[2], -1)
            accuracy = (
                flat_correct[..., square_mask_flat].sum().item()
                / (
                    correct.shape[0]
                    * correct.shape[1]
                    * correct.shape[2]
                    * selected_square_count
                )
            )
            square_num_correct += correct[0].sum(dim=(0, 1)).cpu().to(torch.float64)
            square_num_total += (
                square_mask.cpu().to(torch.float64) * (correct.shape[1] * correct.shape[2])
            )

            weight = batch_end - batch_start
            total_weight += weight
            total_loss += loss.item() * weight
            total_accuracy += accuracy * weight
            flat_preds = preds.reshape(preds.shape[0], preds.shape[1], preds.shape[2], -1)
            flat_targets = targets.reshape(
                targets.shape[0], targets.shape[1], targets.shape[2], -1
            )
            all_preds.append(flat_preds[..., square_mask_flat].reshape(-1).cpu())
            all_targets.append(flat_targets[..., square_mask_flat].reshape(-1).cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    square_accuracy = torch.full((BOARD_ROWS, BOARD_COLS), float("nan"), dtype=torch.float64)
    included_squares = square_num_total > 0
    square_accuracy[included_squares] = (
        square_num_correct[included_squares] / square_num_total[included_squares]
    )
    return {
        "loss": total_loss / total_weight,
        "accuracy": total_accuracy / total_weight,
        "macro_f1": macro_f1_score(preds, targets, num_classes=BOARD_LABEL_OPTIONS),
        "square_accuracy": square_accuracy.tolist(),
    }


def train(config: ProbeConfig) -> dict[int, dict[str, float]]:
    print("Training config:")
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
            "load_weights": not config.random_init,
            "n_head": config.n_head,
        }
    )
    if config.random_init:
        print(
            f"Using randomly initialized transformer with architecture inferred from {config.checkpoint}"
        )
    convert_to_hooked_model(model)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_probe_dataset(
        data_path=config.data_path,
        block_size=model.get_block_size(),
        valid_size=config.valid_size,
        test_size=config.test_size,
        seed=config.seed,
        max_games=config.max_games,
    )
    print(
        "Dataset sizes:",
        json.dumps(
            {
                "train": len(split["train_raw"]),
                "valid": len(split["valid_raw"]),
                "test": len(split["test_raw"]),
            },
            indent=2,
        ),
    )

    pos_start, pos_end = resolve_position_slice(
        config.pos_start,
        config.pos_end,
        model.get_block_size(),
    )
    layers = select_layers(config, model)
    train_tokens = split["train_tokens"]
    train_raw = split["train_raw"]

    results = {}
    for layer in layers:
        probe_name = f"resid_{layer}_linear"
        module_name = f"blocks.{layer}.hook_resid_post"
        print(f"Training layer {layer}")

        probe = torch.randn(
            1,
            model.config.n_embd,
            BOARD_ROWS,
            BOARD_COLS,
            BOARD_LABEL_OPTIONS,
            device=device,
        ) / math.sqrt(model.config.n_embd)
        probe = torch.nn.Parameter(probe)
        optimizer = torch.optim.AdamW(
            [probe], lr=config.lr, betas=(0.9, 0.99), weight_decay=config.wd
        )

        best_valid_loss = float("inf")
        patience = 0
        seen = 0
        stop = False

        for epoch in range(config.num_epochs):
            if stop:
                break

            epoch_indices = torch.randperm(len(train_raw))
            progress = tqdm(
                range(0, len(train_raw), config.batch_size),
                desc=f"layer {layer} epoch {epoch}",
            )
            for batch_start in progress:
                batch_end = min(batch_start + config.batch_size, len(train_raw))
                batch_indices = epoch_indices[batch_start:batch_end].tolist()
                batch_tokens = train_tokens[batch_indices].to(device)
                batch_raw = [train_raw[idx] for idx in batch_indices]

                state_stack = build_state_stack(batch_raw)[:, pos_start:pos_end].to(
                    device
                )
                state_stack_one_hot = state_stack_to_one_hot_threeway(state_stack)

                with torch.no_grad():
                    with record_activations(model, [module_name]) as cache:
                        model(batch_tokens[:, :-1])
                resid_post = (
                    cache[module_name][0][:, pos_start:pos_end].to(device).clone()
                )

                probe_logits = compute_probe_logits(resid_post, probe)
                loss = compute_probe_loss(probe_logits, state_stack_one_hot)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                seen += batch_end - batch_start
                progress.set_postfix(loss=float(loss.item()))

                batch_id = batch_start // config.batch_size
                if batch_id % config.valid_every == 0:
                    metrics = evaluate_probe(
                        model=model,
                        probe=probe.detach(),
                        layer=layer,
                        games_tokens=split["valid_tokens"],
                        games_raw=split["valid_raw"],
                        batch_size=config.batch_size,
                        pos_start=pos_start,
                        pos_end=pos_end,
                        device=device,
                    )
                    print(
                        f"  Validation loss={metrics['loss']:.4f} "
                        f"accuracy={metrics['accuracy']:.4f} "
                        f"macro_f1={metrics['macro_f1']:.4f}"
                    )
                    print("  Validation accuracy by square:")
                    print(
                        format_square_accuracy_board(
                            torch.tensor(metrics["square_accuracy"])
                        )
                    )

                    if metrics["loss"] < best_valid_loss:
                        best_valid_loss = metrics["loss"]
                        patience = 0
                        torch.save(
                            {
                                "probe": probe.detach().cpu(),
                                "layer": layer,
                                "metrics": metrics,
                                "config": asdict(config),
                                "module_name": module_name,
                            },
                            output_dir / f"{probe_name}.pth",
                        )
                    else:
                        patience += 1
                        print(f"  Patience {patience}/{config.valid_patience}")
                        if patience >= config.valid_patience:
                            print(
                                f"  Early stopping layer {layer} after seeing {seen} games"
                            )
                            stop = True
                            break

        test_metrics = evaluate_saved_probe(
            model=model,
            probe_path=output_dir / f"{probe_name}.pth",
            games_tokens=split["test_tokens"],
            games_raw=split["test_raw"],
            batch_size=config.batch_size,
            pos_start=pos_start,
            pos_end=pos_end,
            device=device,
        )
        print(
            f"  Test loss={test_metrics['loss']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        results[layer] = test_metrics

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file_p:
        json.dump(results, file_p, indent=2)
    return results


def evaluate_saved_probe(
    model,
    probe_path: str | Path,
    games_tokens: Tensor,
    games_raw: Sequence[Sequence[int]],
    batch_size: int,
    pos_start: int,
    pos_end: int,
    device: torch.device,
    exclude_center_squares: bool = False,
) -> dict[str, float]:
    artifact = torch.load(probe_path, map_location=device)
    probe = artifact["probe"].to(device)
    layer = int(artifact["layer"])
    return evaluate_probe(
        model=model,
        probe=probe,
        layer=layer,
        games_tokens=games_tokens,
        games_raw=games_raw,
        batch_size=batch_size,
        pos_start=pos_start,
        pos_end=pos_end,
        device=device,
        exclude_center_squares=exclude_center_squares,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=ProbeConfig.checkpoint)
    parser.add_argument(
        "--data-path",
        default=ProbeConfig.data_path,
        help="Path to a pickle dataset file or directory of pickle shards.",
    )
    parser.add_argument("--output-dir", default=ProbeConfig.output_dir)
    parser.add_argument("--device", default=ProbeConfig.device)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help=(
            "Use a randomly initialized transformer with architecture inferred "
            "from --checkpoint instead of loading pretrained weights."
        ),
    )
    parser.add_argument("--n-head", type=int, default=ProbeConfig.n_head)
    parser.add_argument("--lr", type=float, default=ProbeConfig.lr)
    parser.add_argument("--wd", type=float, default=ProbeConfig.wd)
    parser.add_argument("--batch-size", type=int, default=ProbeConfig.batch_size)
    parser.add_argument("--valid-every", type=int, default=ProbeConfig.valid_every)
    parser.add_argument("--pos-start", type=int, default=ProbeConfig.pos_start)
    parser.add_argument(
        "--pos-end",
        type=int,
        default=ProbeConfig.pos_end,
        help="Exclusive end position for the probed move slice; defaults to block_size.",
    )
    parser.add_argument("--num-epochs", type=int, default=ProbeConfig.num_epochs)
    parser.add_argument("--valid-size", type=int, default=ProbeConfig.valid_size)
    parser.add_argument("--test-size", type=int, default=ProbeConfig.test_size)
    parser.add_argument(
        "--valid-patience", type=int, default=ProbeConfig.valid_patience
    )
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--train-layers", type=str, default=ProbeConfig.train_layers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProbeConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        random_init=args.random_init,
        n_head=args.n_head,
        lr=args.lr,
        wd=args.wd,
        batch_size=args.batch_size,
        valid_every=args.valid_every,
        pos_start=args.pos_start,
        pos_end=args.pos_end,
        num_epochs=args.num_epochs,
        valid_size=args.valid_size,
        test_size=args.test_size,
        valid_patience=args.valid_patience,
        seed=args.seed,
        max_games=args.max_games,
        max_layers=args.max_layers,
        train_layers=args.train_layers,
    )
    train(config)


if __name__ == "__main__":
    main()
