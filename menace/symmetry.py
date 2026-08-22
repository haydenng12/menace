"""Board symmetry helpers for MENACE.

The tic-tac-toe board has 8 symmetries: 4 rotations and 4 reflections.
MENACE stores only the lexicographically smallest transformed board as the
canonical matchbox key. Moves are transformed into and back out of that
canonical orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import board_key

# Each transform maps an ORIGINAL index -> TRANSFORMED index.
TRANSFORMS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),  # identity
    (2, 5, 8, 1, 4, 7, 0, 3, 6),  # rotate 90 clockwise
    (8, 7, 6, 5, 4, 3, 2, 1, 0),  # rotate 180
    (6, 3, 0, 7, 4, 1, 8, 5, 2),  # rotate 270 clockwise
    (2, 1, 0, 5, 4, 3, 8, 7, 6),  # reflect vertical axis
    (6, 7, 8, 3, 4, 5, 0, 1, 2),  # reflect horizontal axis
    (0, 3, 6, 1, 4, 7, 2, 5, 8),  # reflect main diagonal
    (8, 5, 2, 7, 4, 1, 6, 3, 0),  # reflect anti-diagonal
)


@dataclass(frozen=True)
class CanonicalBoard:
    key: str
    transform: tuple[int, ...]

    def move_to_canonical(self, move: int) -> int:
        return self.transform[move]

    def move_from_canonical(self, move: int) -> int:
        return self.transform.index(move)


def transform_board(board: list[str], transform: tuple[int, ...]) -> list[str]:
    transformed = [" "] * 9
    for original_index, transformed_index in enumerate(transform):
        transformed[transformed_index] = board[original_index]
    return transformed


def canonicalize(board: list[str]) -> CanonicalBoard:
    candidates = [
        (board_key(transform_board(board, transform)), transform)
        for transform in TRANSFORMS
    ]
    key, transform = min(candidates, key=lambda item: item[0])
    return CanonicalBoard(key=key, transform=transform)
