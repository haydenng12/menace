"""Command-line entry point for the MENACE research experiments."""

import argparse

from menace.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--train-games", type=int, default=5000)
    parser.add_argument("--eval-games", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_experiment(args.output, args.trials, args.train_games, args.eval_games, args.checkpoint_every, args.seed)


if __name__ == "__main__":
    main()
