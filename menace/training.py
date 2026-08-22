"""Training helpers for MENACE."""

from __future__ import annotations

from dataclasses import dataclass

from .game import O, X, play_game
from .players import MenacePlayer, MinimaxPlayer, RandomPlayer


@dataclass
class TrainingStats:
    games: int
    wins: int
    draws: int
    losses: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games if self.games else 0.0


def train_against_random(
    menace: MenacePlayer,
    games: int = 50_000,
    menace_symbol: str = X,
    report_every: int = 5_000,
) -> TrainingStats:
    random_player = RandomPlayer()
    wins = draws = losses = 0

    for game_num in range(1, games + 1):
        menace.reset_history()

        if menace_symbol == X:
            result = play_game(menace, random_player)
        else:
            result = play_game(random_player, menace)

        if result.winner == menace_symbol:
            wins += 1
            menace.learn("win")
        elif result.winner is None:
            draws += 1
            menace.learn("draw")
        else:
            losses += 1
            menace.learn("loss")

        if report_every and game_num % report_every == 0:
            print(
                f"{game_num:>7,} games | "
                f"wins {wins/game_num:6.1%} | "
                f"draws {draws/game_num:6.1%} | "
                f"losses {losses/game_num:6.1%}"
            )

    return TrainingStats(games, wins, draws, losses)



def train_self_play(
    player_x: MenacePlayer,
    player_o: MenacePlayer,
    games: int = 50_000,
    report_every: int = 5_000,
) -> TrainingStats:
    """Train two independent MENACE players against each other."""
    x_wins = draws = o_wins = 0

    for game_num in range(1, games + 1):
        player_x.reset_history()
        player_o.reset_history()
        result = play_game(player_x, player_o)

        if result.winner == X:
            x_wins += 1
            player_x.learn("win")
            player_o.learn("loss")
        elif result.winner == O:
            o_wins += 1
            player_x.learn("loss")
            player_o.learn("win")
        else:
            draws += 1
            player_x.learn("draw")
            player_o.learn("draw")

        if report_every and game_num % report_every == 0:
            print(
                f"{game_num:>7,} games | "
                f"X wins {x_wins/game_num:6.1%} | "
                f"draws {draws/game_num:6.1%} | "
                f"O wins {o_wins/game_num:6.1%}"
            )

    return TrainingStats(games, x_wins, draws, o_wins)


def evaluate_against_minimax(
    menace: MenacePlayer,
    games: int = 1_000,
    menace_symbol: str = X,
) -> TrainingStats:
    """Evaluate MENACE without learning against a perfect minimax player."""
    minimax = MinimaxPlayer()
    wins = draws = losses = 0

    for _ in range(games):
        menace.reset_history()
        if menace_symbol == X:
            result = play_game(menace, minimax)
        else:
            result = play_game(minimax, menace)

        if result.winner == menace_symbol:
            wins += 1
        elif result.winner is None:
            draws += 1
        else:
            losses += 1

    menace.reset_history()
    return TrainingStats(games, wins, draws, losses)
