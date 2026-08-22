"""Players for tic-tac-toe, including the MENACE learning agent."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .game import EMPTY, O, X, board_from_key, board_key, is_draw, make_move, valid_moves, winner
from .symmetry import canonicalize


class RandomPlayer:
    def get_move(self, board: list[str], symbol: str) -> int:
        return random.choice(valid_moves(board))

    def learn(self, result: str) -> None:
        return None


class MinimaxPlayer:
    """Perfect tic-tac-toe player used to evaluate MENACE."""

    def get_move(self, board: list[str], symbol: str) -> int:
        moves = valid_moves(board)
        if not moves:
            raise ValueError("no valid moves available")

        opponent = O if symbol == X else X
        best_score = -10
        best_move = moves[0]

        for move in moves:
            next_board = make_move(board, move, symbol)
            score = self._minimax(next_board, opponent, symbol)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, board: list[str], current: str, maximizing_symbol: str) -> int:
        win = winner(board)
        if win == maximizing_symbol:
            return 1
        if win is not None:
            return -1
        if is_draw(board):
            return 0

        opponent = O if current == X else X
        scores = [
            self._minimax(make_move(board, move, current), opponent, maximizing_symbol)
            for move in valid_moves(board)
        ]

        if current == maximizing_symbol:
            return max(scores)
        return min(scores)

    def learn(self, result: str) -> None:
        return None


class MenacePlayer:
    """A simple MENACE matchbox learner.

    Each board state maps to a matchbox. Each legal move has a bead count.
    Moves are sampled with probability proportional to their bead counts.
    After a game, bead counts are reinforced or punished.
    """

    def __init__(
        self,
        initial_beads: int = 3,
        win_reward: int = 3,
        draw_reward: int = 1,
        loss_penalty: int = 1,
    ) -> None:
        if initial_beads < 1:
            raise ValueError("initial_beads must be at least 1")
        self.initial_beads = initial_beads
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_penalty = loss_penalty
        self.matchboxes: dict[str, dict[int, int]] = {}
        self.game_history: list[tuple[str, int]] = []

    def reset_history(self) -> None:
        self.game_history.clear()

    def _canonical_matchbox(
        self, board: list[str]
    ) -> tuple[str, dict[int, int], tuple[int, ...]]:
        canonical = canonicalize(board)
        canonical_board = board_from_key(canonical.key)

        if canonical.key not in self.matchboxes:
            self.matchboxes[canonical.key] = {
                move: self.initial_beads for move in valid_moves(canonical_board)
            }

        return canonical.key, self.matchboxes[canonical.key], canonical.transform

    def _ensure_matchbox(self, board: list[str]) -> dict[int, int]:
        """Return the shared canonical matchbox for this board.

        Kept as a small compatibility helper for tests and external users.
        The dictionary's move indices are in canonical-board coordinates.
        """
        _, box, _ = self._canonical_matchbox(board)
        return box

    def get_move(self, board: list[str], symbol: str) -> int:
        key, box, transform = self._canonical_matchbox(board)
        canonical_moves = list(box.keys())
        weights = [box[m] for m in canonical_moves]

        # Never allow a state to become permanently unplayable.
        if sum(weights) <= 0:
            for move in canonical_moves:
                box[move] = self.initial_beads
            weights = [box[m] for m in canonical_moves]

        canonical_move = random.choices(canonical_moves, weights=weights, k=1)[0]

        # Learning history is stored entirely in canonical coordinates.
        self.game_history.append((key, canonical_move))

        # Transform the chosen canonical square back to the board orientation
        # that the game engine gave us.
        original_move = transform.index(canonical_move)
        return original_move

    def learn(self, result: str) -> None:
        if result not in {"win", "draw", "loss"}:
            raise ValueError("result must be 'win', 'draw', or 'loss'")

        delta = {
            "win": self.win_reward,
            "draw": self.draw_reward,
            "loss": -self.loss_penalty,
        }[result]

        for key, move in self.game_history:
            current = self.matchboxes[key][move]
            self.matchboxes[key][move] = max(1, current + delta)

        self.reset_history()

    def save(self, path: str | Path) -> None:
        data = {
            "initial_beads": self.initial_beads,
            "win_reward": self.win_reward,
            "draw_reward": self.draw_reward,
            "loss_penalty": self.loss_penalty,
            "matchboxes": {
                key: {str(move): beads for move, beads in moves.items()}
                for key, moves in self.matchboxes.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MenacePlayer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        player = cls(
            initial_beads=data["initial_beads"],
            win_reward=data["win_reward"],
            draw_reward=data["draw_reward"],
            loss_penalty=data["loss_penalty"],
        )
        # Fold saved states into canonical symmetry classes. This also makes
        # older models from pre-symmetry versions load correctly.
        for key, moves in data["matchboxes"].items():
            board = board_from_key(key)
            canonical = canonicalize(board)
            target = player.matchboxes.setdefault(canonical.key, {})

            for move_text, beads in moves.items():
                original_move = int(move_text)
                canonical_move = canonical.move_to_canonical(original_move)
                target[canonical_move] = target.get(canonical_move, 0) + int(beads)

        return player
