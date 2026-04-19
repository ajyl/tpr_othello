"""Helpers for loading one or more pickle datasets from a file or directory."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PICKLE_SUFFIXES = (".pickle", ".pkl")


def resolve_pickle_paths(data_path: str | Path) -> list[Path]:
    path = Path(data_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        pickle_paths = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix in PICKLE_SUFFIXES
        )
        if not pickle_paths:
            raise FileNotFoundError(f"No pickle files found in dataset directory: {path}")
        return pickle_paths
    if path.exists():
        raise ValueError(f"Expected a pickle file or directory, got: {path}")
    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def load_pickle_sequences(data_path: str | Path) -> list[Any]:
    merged_sequences = []
    for pickle_path in resolve_pickle_paths(data_path):
        with pickle_path.open("rb") as file_p:
            data = pickle.load(file_p)
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(
                f"Expected a sequence dataset in {pickle_path}, got {type(data)!r}"
            )
        merged_sequences.extend(data)

    if not merged_sequences:
        raise ValueError(f"Dataset is empty: {data_path}")
    return merged_sequences
