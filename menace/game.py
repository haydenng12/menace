"""Tic-tac-toe game utilities used by MENACE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

EMPTY = " "
X = "X"
O = "O"
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


def new_board() -> list[str]:
    return [EMPTY] * 9


def valid_moves(board: list[str]) -> list[int]:
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def make_move(board: list[str], move: int, symbol: str) -> list[str]:
    if move not in range(9):
        raise ValueError("move must be between 0 and 8")
    if board[move] != EMPTY:
        raise ValueError("cell is already occupied")
    next_board = board.copy()
    next_board[move] = symbol
    return next_board


def winner(board: list[str]) -> str | None:
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: list[str]) -> bool:
    return winner(board) is None and not valid_moves(board)


def board_key(board: list[str]) -> str:
    return "".join("-" if cell == EMPTY else cell for cell in board)


def board_from_key(key: str) -> list[str]:
    if len(key) != 9:
        raise ValueError("board key must contain exactly 9 characters")
    return [EMPTY if ch == "-" else ch for ch in key]


def render(board: list[str], show_positions: bool = False) -> str:
    cells: list[str] = []
    for i, cell in enumerate(board):
        if cell == EMPTY and show_positions:
            cells.append(str(i + 1))
        else:
            cells.append(cell)
    return (
        f" {cells[0]} | {cells[1]} | {cells[2]} \n"
        "---+---+---\n"
        f" {cells[3]} | {cells[4]} | {cells[5]} \n"
        "---+---+---\n"
        f" {cells[6]} | {cells[7]} | {cells[8]} "
    )


@dataclass(frozen=True)
class GameResult:
    winner: str | None
    moves: int


def play_game(player_x, player_o) -> GameResult:
    board = new_board()
    current = X
    moves = 0

    while True:
        player = player_x if current == X else player_o
        move = player.get_move(board.copy(), current)
        board = make_move(board, move, current)
        moves += 1

        win = winner(board)
        if win is not None:
            return GameResult(win, moves)
        if is_draw(board):
            return GameResult(None, moves)
        current = O if current == X else X
