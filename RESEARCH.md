# MENACE Experimental Research

## Research question

How do reward structure, explicit exploration, symmetry reduction, training opponent, and matchbox initialization affect the learning efficiency and final policy of MENACE-style reinforcement learning?

## Experimental design

The baseline is the repository's original MENACE design: three initial beads per legal action, rewards of +3/+1/-1 for win/draw/loss, bead-proportional action sampling, symmetry reduction, and training against a random opponent. Each treatment changes exactly one factor. The study runs 100 independently seeded trials per condition rather than treating one long run as independent evidence.

All agents are trained as X. At each checkpoint, the current policy is evaluated without learning against an independently seeded random opponent. Final comparisons use the same opponent class and number of games for every condition, including agents trained through self-play or against minimax.

### Outcomes

- **Win, draw, and loss rates:** held-out final evaluation.
- **Variance and 95% confidence interval:** uncertainty across independent training trials.
- **Games to convergence:** first checkpoint after which validation score (`win rate + 0.5 * draw rate`) never falls more than 0.05 below its final value.
- **State-space size:** number of matchboxes created by the trained X agent.
- **Policy stability:** fraction of shared matchboxes whose deterministic greedy action is unchanged between the final two checkpoints.

See `results/metadata.json` for the complete configuration and operational definitions.

## Reproduce

Quick smoke test:

```bash
python run_experiments.py --trials 2 --train-games 100 --eval-games 20 --checkpoint-every 50 --output smoke-results
```

Full experiment used in the paper:

```bash
python run_experiments.py --trials 100 --train-games 3000 --eval-games 200 --checkpoint-every 300 --seed 20260910 --output results
python make_figures.py
```

Automated tests:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Research artifacts

- `menace/experiments.py`: conditions, trial runner, metrics, summary statistics
- `results/trial_results.csv`: all 1,100 trial-level observations
- `results/summary.csv`: means, sample variances, and 95% confidence intervals
- `results/metadata.json`: seeds, sample sizes, controls, metric definitions
- `results/figures/`: publication figures
- `paper/menace_research_paper.pdf`: 4-6 page mini-paper
