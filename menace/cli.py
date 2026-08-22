"""Command-line interface for training and playing MENACE."""

from __future__ import annotations

import argparse
from pathlib import Path

from .game import O, X, is_draw, make_move, new_board, render, winner
from .players import MenacePlayer
from .training import evaluate_against_minimax, train_against_random, train_self_play

DEFAULT_MODEL = Path("menace_model.json")


def train_command(args: argparse.Namespace) -> None:
    menace = MenacePlayer(
        initial_beads=args.initial_beads,
        win_reward=args.win_reward,
        draw_reward=args.draw_reward,
        loss_penalty=args.loss_penalty,
    )
    stats = train_against_random(
        menace,
        games=args.games,
        menace_symbol=X,
        report_every=args.report_every,
    )
    menace.save(args.output)
    print(f"\nSaved model to {args.output}")
    print(
        f"Final: {stats.wins} wins, {stats.draws} draws, "
        f"{stats.losses} losses ({stats.win_rate:.1%} win rate)"
    )


def play_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if model_path.exists():
        menace = MenacePlayer.load(model_path)
    else:
        print(f"No model found at {model_path}; using an untrained MENACE.")
        menace = MenacePlayer()

    human = X if args.first else O
    bot = O if human == X else X
    board = new_board()
    current = X
    menace.reset_history()

    print("\nBoard positions:")
    print(render(board, show_positions=True))

    while True:
        print("\n" + render(board, show_positions=True))
        if current == human:
            while True:
                try:
                    move = int(input("Your move (1-9): ")) - 1
                    board = make_move(board, move, human)
                    break
                except (ValueError, IndexError):
                    print("Choose an empty square from 1 to 9.")
        else:
            move = menace.get_move(board.copy(), bot)
            board = make_move(board, move, bot)
            print(f"MENACE chooses {move + 1}.")

        win = winner(board)
        if win is not None:
            print("\n" + render(board))
            if win == human:
                print("You win!")
                menace.learn("loss")
            else:
                print("MENACE wins!")
                menace.learn("win")
            break

        if is_draw(board):
            print("\n" + render(board))
            print("Draw!")
            menace.learn("draw")
            break

        current = O if current == X else X

    if args.learn:
        menace.save(model_path)
        print(f"Updated model saved to {model_path}.")



def self_play_command(args: argparse.Namespace) -> None:
    player_x = MenacePlayer(
        initial_beads=args.initial_beads,
        win_reward=args.win_reward,
        draw_reward=args.draw_reward,
        loss_penalty=args.loss_penalty,
    )
    player_o = MenacePlayer(
        initial_beads=args.initial_beads,
        win_reward=args.win_reward,
        draw_reward=args.draw_reward,
        loss_penalty=args.loss_penalty,
    )
    stats = train_self_play(
        player_x,
        player_o,
        games=args.games,
        report_every=args.report_every,
    )
    player_x.save(args.x_output)
    player_o.save(args.o_output)
    print(f"\nSaved X model to {args.x_output}")
    print(f"Saved O model to {args.o_output}")
    print(
        f"Final: X wins {stats.wins}, draws {stats.draws}, "
        f"O wins {stats.losses}"
    )


def compare_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    menace = MenacePlayer.load(model_path)

    for symbol in (X, O):
        stats = evaluate_against_minimax(
            menace,
            games=args.games,
            menace_symbol=symbol,
        )
        print(
            f"MENACE as {symbol}: "
            f"{stats.wins} wins, {stats.draws} draws, {stats.losses} losses "
            f"| win {stats.win_rate:.1%}, loss {stats.loss_rate:.1%}"
        )


def visual_command(args: argparse.Namespace) -> None:
    from .visual import launch_visualizer
    launch_visualizer()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="menace",
        description="Train and play against MENACE, a matchbox-learning tic-tac-toe AI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train MENACE against a random player")
    train.add_argument("--games", type=int, default=50_000)
    train.add_argument("--initial-beads", type=int, default=3)
    train.add_argument("--report-every", type=int, default=5_000)
    train.add_argument("--output", default=str(DEFAULT_MODEL))
    train.add_argument("--win-reward", type=int, default=3)
    train.add_argument("--draw-reward", type=int, default=1)
    train.add_argument("--loss-penalty", type=int, default=1)
    train.set_defaults(func=train_command)

    play = subparsers.add_parser("play", help="play tic-tac-toe against MENACE")
    play.add_argument("--model", default=str(DEFAULT_MODEL))
    play.add_argument("--first", action="store_true", help="play first as X")
    play.add_argument(
        "--learn",
        action="store_true",
        help="save reinforcement from your game back into the model",
    )
    play.set_defaults(func=play_command)


    self_play = subparsers.add_parser(
        "self-play",
        help="train two MENACE players against each other",
    )
    self_play.add_argument("--games", type=int, default=50_000)
    self_play.add_argument("--initial-beads", type=int, default=3)
    self_play.add_argument("--win-reward", type=int, default=3)
    self_play.add_argument("--draw-reward", type=int, default=1)
    self_play.add_argument("--loss-penalty", type=int, default=1)
    self_play.add_argument("--report-every", type=int, default=5_000)
    self_play.add_argument("--x-output", default="menace_x.json")
    self_play.add_argument("--o-output", default="menace_o.json")
    self_play.set_defaults(func=self_play_command)

    compare = subparsers.add_parser(
        "compare",
        help="compare a trained MENACE model against perfect minimax",
    )
    compare.add_argument("--model", default=str(DEFAULT_MODEL))
    compare.add_argument("--games", type=int, default=1_000)
    compare.set_defaults(func=compare_command)

    visual = subparsers.add_parser("visual", help="open the visual MENACE training dashboard")
    visual.set_defaults(func=visual_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
