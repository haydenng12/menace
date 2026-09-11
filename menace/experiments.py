"""Reproducible controlled experiments for MENACE.

Every condition changes one factor from the baseline. Training and evaluation
use separate seeded random streams, and evaluation never updates matchboxes.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .game import O, X, play_game
from .players import MenacePlayer, MinimaxPlayer, RandomPlayer


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    initial_beads: int = 3
    win_reward: int = 3
    draw_reward: int = 1
    loss_penalty: int = 1
    epsilon: float = 0.0
    symmetry: bool = True
    opponent: str = "random"


@dataclass
class TrialResult:
    condition: str
    family: str
    trial: int
    seed: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    games_to_convergence: int
    state_space_size: int
    policy_stability: float


BASELINE = Condition("baseline", "baseline")
CONDITIONS = (
    BASELINE,
    replace(BASELINE, name="reward_win_heavy", family="reward", win_reward=5),
    replace(BASELINE, name="reward_no_draw", family="reward", draw_reward=0),
    replace(BASELINE, name="reward_harsh_loss", family="reward", loss_penalty=3),
    replace(BASELINE, name="exploration_epsilon_0.10", family="exploration", epsilon=0.10),
    replace(BASELINE, name="exploration_epsilon_0.30", family="exploration", epsilon=0.30),
    replace(BASELINE, name="symmetry_off", family="symmetry", symmetry=False),
    replace(BASELINE, name="opponent_self_play", family="opponent", opponent="self_play"),
    replace(BASELINE, name="opponent_minimax", family="opponent", opponent="minimax"),
    replace(BASELINE, name="initial_beads_1", family="initialization", initial_beads=1),
    replace(BASELINE, name="initial_beads_10", family="initialization", initial_beads=10),
)


def _agent(condition: Condition, seed: int) -> MenacePlayer:
    return MenacePlayer(
        initial_beads=condition.initial_beads,
        win_reward=condition.win_reward,
        draw_reward=condition.draw_reward,
        loss_penalty=condition.loss_penalty,
        use_symmetry=condition.symmetry,
        exploration_epsilon=condition.epsilon,
        rng=random.Random(seed),
    )


def _outcome(agent: MenacePlayer, opponent, learn_opponent: bool = False) -> None:
    agent.reset_history()
    if learn_opponent:
        opponent.reset_history()
    result = play_game(agent, opponent)
    if result.winner == X:
        agent.learn("win")
        if learn_opponent:
            opponent.learn("loss")
    elif result.winner is None:
        agent.learn("draw")
        if learn_opponent:
            opponent.learn("draw")
    else:
        agent.learn("loss")
        if learn_opponent:
            opponent.learn("win")


def _evaluate(agent: MenacePlayer, games: int, seed: int) -> tuple[float, float, float]:
    """Evaluate as X against random while preserving the training RNG state."""
    rng_state = agent.rng.getstate()
    old_epsilon = agent.exploration_epsilon
    agent.exploration_epsilon = 0.0
    opponent = RandomPlayer()
    random_state = random.getstate()
    random.seed(seed)
    wins = draws = losses = 0
    try:
        for _ in range(games):
            agent.reset_history()
            result = play_game(agent, opponent)
            if result.winner == X:
                wins += 1
            elif result.winner is None:
                draws += 1
            else:
                losses += 1
    finally:
        agent.reset_history()
        agent.rng.setstate(rng_state)
        agent.exploration_epsilon = old_epsilon
        random.setstate(random_state)
    return wins / games, draws / games, losses / games


def _policy(agent: MenacePlayer) -> dict[str, int]:
    return {
        key: min(move for move, beads in moves.items() if beads == max(moves.values()))
        for key, moves in agent.matchboxes.items()
    }


def _stability(previous: dict[str, int], current: dict[str, int]) -> float:
    shared = previous.keys() & current.keys()
    return sum(previous[k] == current[k] for k in shared) / len(shared) if shared else 0.0


def _convergence(checkpoints: list[int], scores: list[float], tolerance: float = 0.05) -> int:
    """First checkpoint after which score never falls > tolerance below final score."""
    floor = scores[-1] - tolerance
    for index, game in enumerate(checkpoints):
        if all(score >= floor for score in scores[index:]):
            return game
    return checkpoints[-1]


def run_trial(
    condition: Condition,
    trial: int,
    base_seed: int,
    train_games: int,
    eval_games: int,
    checkpoint_every: int,
) -> TrialResult:
    seed = base_seed + trial
    agent = _agent(condition, seed)
    if condition.opponent == "random":
        opponent = RandomPlayer()
        learn_opponent = False
    elif condition.opponent == "minimax":
        opponent = MinimaxPlayer()
        learn_opponent = False
    elif condition.opponent == "self_play":
        opponent = _agent(condition, seed + 10_000_000)
        learn_opponent = True
    else:
        raise ValueError(f"unknown opponent: {condition.opponent}")

    random_state = random.getstate()
    random.seed(seed + 20_000_000)
    checkpoints: list[int] = []
    scores: list[float] = []
    policies: list[dict[str, int]] = []
    try:
        for game in range(1, train_games + 1):
            _outcome(agent, opponent, learn_opponent)
            if game % checkpoint_every == 0 or game == train_games:
                win, draw, _ = _evaluate(agent, eval_games, seed + 30_000_000 + game)
                checkpoints.append(game)
                scores.append(win + 0.5 * draw)
                policies.append(_policy(agent))
    finally:
        random.setstate(random_state)

    win, draw, loss = _evaluate(agent, eval_games * 2, seed + 40_000_000)
    stability = _stability(policies[-2], policies[-1]) if len(policies) > 1 else 0.0
    return TrialResult(
        condition=condition.name,
        family=condition.family,
        trial=trial,
        seed=seed,
        win_rate=win,
        draw_rate=draw,
        loss_rate=loss,
        games_to_convergence=_convergence(checkpoints, scores),
        state_space_size=len(agent.matchboxes),
        policy_stability=stability,
    )


def _summary(results: list[TrialResult]) -> list[dict[str, float | str | int]]:
    rows = []
    for condition in CONDITIONS:
        group = [r for r in results if r.condition == condition.name]
        if not group:
            continue
        row: dict[str, float | str | int] = {
            "condition": condition.name,
            "family": condition.family,
            "trials": len(group),
        }
        for field in ("win_rate", "draw_rate", "loss_rate", "games_to_convergence", "state_space_size", "policy_stability"):
            values = [float(getattr(r, field)) for r in group]
            mean = statistics.mean(values)
            variance = statistics.variance(values) if len(values) > 1 else 0.0
            se = math.sqrt(variance / len(values)) if values else 0.0
            row[f"mean_{field}"] = mean
            row[f"variance_{field}"] = variance
            row[f"ci95_low_{field}"] = mean - 1.96 * se
            row[f"ci95_high_{field}"] = mean + 1.96 * se
        rows.append(row)
    return rows


def _paired_comparisons(results: list[TrialResult]) -> list[dict[str, float | str]]:
    """Paired treatment-minus-baseline effects using matching trial seeds."""
    baseline = {r.trial: r for r in results if r.condition == "baseline"}
    rows = []
    for condition in CONDITIONS:
        if condition.name == "baseline":
            continue
        group = {r.trial: r for r in results if r.condition == condition.name}
        differences = [group[i].win_rate - baseline[i].win_rate for i in sorted(baseline.keys() & group.keys())]
        mean = statistics.mean(differences)
        variance = statistics.variance(differences) if len(differences) > 1 else 0.0
        margin = 1.96 * math.sqrt(variance / len(differences))
        rows.append({"condition": condition.name, "mean_win_rate_difference": mean, "ci95_low": mean-margin, "ci95_high": mean+margin})
    return rows


def run_experiment(
    output_dir: str | Path = "results",
    trials: int = 100,
    train_games: int = 5_000,
    eval_games: int = 300,
    checkpoint_every: int = 500,
    base_seed: int = 20260910,
    conditions: tuple[Condition, ...] = CONDITIONS,
) -> list[dict[str, float | str | int]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results: list[TrialResult] = []
    for condition in conditions:
        print(f"Running {condition.name} ({trials} independent trials)")
        for trial in range(trials):
            all_results.append(run_trial(condition, trial, base_seed, train_games, eval_games, checkpoint_every))

    trial_rows = [asdict(result) for result in all_results]
    summary = _summary(all_results)
    _write_csv(output / "trial_results.csv", trial_rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "paired_comparisons.csv", _paired_comparisons(all_results))
    metadata = {
        "trials_per_condition": trials,
        "train_games_per_trial": train_games,
        "evaluation_games_per_checkpoint": eval_games,
        "final_evaluation_games": eval_games * 2,
        "checkpoint_every": checkpoint_every,
        "base_seed": base_seed,
        "convergence_definition": "first checkpoint after which validation score (win + 0.5*draw) never falls more than 0.05 below final score",
        "policy_stability_definition": "fraction of shared states whose deterministic greedy action is unchanged across the final two checkpoints",
        "conditions": [asdict(c) for c in conditions],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
