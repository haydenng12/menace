# From Demonstration to Experiment: Controlled Ablations of MENACE Learning

**Hayden Ng**  
September 2026

## Abstract

MENACE is an early reinforcement-learning system that represents tic-tac-toe states as matchboxes and actions as colored beads. Prior demonstrations typically show that a single agent improves over one long training run, but do not isolate which design choices cause improvement or quantify variability between runs. This study asks how reward structure, explicit exploration, symmetry reduction, training opponent, and matchbox initialization affect MENACE learning. Eleven controlled conditions were evaluated across 100 independently seeded trials each. Every trial used 3,000 training games and a held-out random-opponent benchmark. Symmetry reduction increased mean win rate from 70.23% to 80.30% and reduced the learned state space from 2,118.0 to 336.6 matchboxes. A stronger loss penalty achieved the highest win rate (85.10%), while excessive initial beads and added epsilon exploration reduced performance. These results show that representation and credit assignment materially influence MENACE's sample efficiency, even in a small deterministic environment.

## 1. Introduction

Donald Michie's Matchbox Educable Noughts and Crosses Engine (MENACE) is a physical and computational illustration of learning through reward and punishment. For each encountered board state, a matchbox stores beads corresponding to legal moves. A move is sampled in proportion to its bead count, and the beads associated with a completed game are reinforced or removed according to the outcome.

A working MENACE implementation establishes that learning can occur, but a research claim requires controlled comparisons, repeated independent runs, uncertainty estimates, and operational definitions chosen before inspecting outcomes. This project therefore moves from “Can MENACE learn?” to the following question:

> How do reward structure, explicit exploration, symmetry reduction, training opponent, and initialization affect MENACE's final performance, convergence, memory use, and policy stability?

The central hypothesis was that symmetry reduction would improve sample efficiency and reduce memory without materially restricting the best achievable policy. Secondary hypotheses were that stronger negative feedback would suppress losing actions more quickly, moderate exploration might avoid premature concentration, and large uniform initial bead counts would slow adaptation.

## 2. Background

In reinforcement-learning terminology, the board is the **state**, an empty square is an **action**, and the win/draw/loss update is the **reward signal**. MENACE does not estimate a value function explicitly. Instead, each bead count is a nonnegative action preference. If action *a* has bead count *b(a)* in state *s*, its selection probability is `b(a) / sum b(a')` across legal actions. The lower bound of one bead prevents an action from becoming permanently impossible.

Tic-tac-toe has geometric redundancy. Rotating or reflecting a board does not change its strategic meaning. Canonicalization maps all eight rotations and reflections to one representative state and maps the selected action back to the original orientation. This shares experience across equivalent observations and is the component examined by the primary ablation.

## 3. Methodology

The baseline used three initial beads per action, win/draw/loss updates of +3/+1/-1, bead-proportional sampling, symmetry reduction, and a random training opponent. Ten treatments each changed one factor: three reward variants, two epsilon-exploration levels, symmetry disabled, self-play, minimax training, and initial bead counts of one or ten.

Each condition used 100 independent trials. Trial seeds were deterministic and recorded; training and evaluation used separate pseudorandom streams. Each trial trained MENACE as X for 3,000 games. At 300-game checkpoints, the frozen policy played 200 validation games against a separately seeded random opponent. A final held-out evaluation used 400 games. Evaluation changed neither matchboxes nor the training random-number state. All conditions were evaluated against the same opponent class, including those trained through self-play or against minimax.

The outcomes were final win, draw, and loss rate; sample variance and a normal-approximation 95% confidence interval across trials; games to convergence; state-space size; and policy stability. Validation score was defined as `win rate + 0.5 x draw rate`. Convergence was the first checkpoint after which this score never fell more than 0.05 below its final value. Policy stability was the fraction of shared states whose deterministic greedy action was unchanged between the final two checkpoints, with ties broken by the lowest board index.

## 4. Results

The baseline achieved an 80.30% mean final win rate (95% CI 79.84%-80.75%). Reward design had a clear effect. The harsh-loss condition achieved the best result at 85.10% (84.56%-85.64%) and converged in 1,599 games on average, compared with 1,743 for baseline. Removing the draw reward and increasing the win reward also improved win rate, to 82.78% and 82.47% respectively.

Added epsilon exploration did not help. Epsilon values of 0.10 and 0.30 reduced win rate to 78.94% and 77.48%, and the larger value reduced final policy stability from 95.6% to 93.6%. This is consistent with MENACE already exploring through probabilistic bead sampling; epsilon adds uniform random actions that partially ignore learned preferences.

Initialization mattered strongly within the fixed 3,000-game budget. One initial bead produced 82.30% win rate and 97.0% policy stability. Ten beads produced only 73.09% and 93.3%. A larger prior requires more reward updates before observed outcomes can noticeably alter action probabilities.

Training opponent also changed the learned policy. Self-play slightly exceeded baseline at 81.68% and had the fastest credible high-performing convergence (1,566 games). Minimax-trained MENACE scored only 60.75% against random despite an apparently fast convergence of 864 games. The latter is not evidence of a superior learner: its final validation plateau was lower, so it reached that plateau quickly.

## 5. Symmetry Ablation

Disabling symmetry while holding every other baseline factor fixed reduced mean win rate by 10.07 percentage points, from 80.30% to 70.23%. The 95% confidence intervals do not overlap. Mean state-space size increased from 336.6 to 2,118.0 matchboxes, a 6.29-fold increase; equivalently, canonicalization reduced stored states by 84.1%.

The ablation supports two related mechanisms. First, symmetry reduction saves memory by merging equivalent boards. Second, every update generalizes to rotated and reflected versions, so each canonical matchbox receives more relevant experience. Without reduction, the same number of games is spread across many redundant states. Eventual optimal performance might become similar with much more training, but within the controlled sample budget symmetry materially improves sample efficiency.

Convergence time alone obscures this result: symmetry-off convergence was 1,698 games, slightly faster than baseline's 1,743, because the convergence definition is relative to each condition's own final score. Final performance and convergence must therefore be interpreted together. Policy stability was almost unchanged (95.7% versus 95.6%), showing that a stable policy can still be stably inferior.

## 6. Discussion

The experiments indicate that representation, reward design, and initialization matter more than adding generic exploration. Symmetry is the most consequential systems-level component because it simultaneously reduces storage and lets experience transfer across equivalent states. The strong loss penalty appears to improve credit assignment by removing probability mass from losing trajectories faster. The result does not imply that arbitrarily large penalties are always better: bead counts are clipped at one, and more extreme penalties may behave identically or destabilize learning under another design.

The opponent result illustrates distribution shift. Minimax visits a narrow set of strategically strong responses, whereas the final benchmark includes many random, suboptimal moves. Training only against perfect play does not expose the learner to all states induced by a random opponent. Self-play supplies a broader curriculum that changes as both agents improve, which may explain its competitive result.

## 7. Limitations

Tic-tac-toe is tiny, deterministic, fully observable, and solved, so the findings should not be generalized directly to modern reinforcement learning. MENACE uses tabular preferences rather than neural function approximation. Only MENACE as X was studied, the training budget was fixed at 3,000 games, and final evaluation used a random opponent. Confidence intervals quantify variation between seeded runs but do not eliminate bias from the selected environment, metrics, or condition grid. The convergence statistic is relative to final performance and should not be read as time to an optimal policy. Multiple comparisons were exploratory; no family-wise significance correction was applied.

## 8. Future Work

Future experiments should train for longer horizons to distinguish delayed from permanently weaker conditions, repeat all treatments with MENACE as O, and evaluate against mixtures of random, heuristic, self-play, and minimax opponents. A factorial design could test interactions, especially between symmetry and initialization or reward penalty. Alternative uncertainty methods such as bootstrap intervals and survival-style time-to-threshold analysis would strengthen inference. Finally, MENACE can be compared with Q-learning and Monte Carlo control under matched state representations and sample budgets.

## Reproducibility and References

The repository includes every trial-level observation, summaries, metadata, figures, source code, and tests. The exact full command is documented in `RESEARCH.md`; rerunning it with seed 20260910 reconstructs the experiment.

1. Michie, D. (1961). Trial and error. *Science Survey*, Part 2, 129-145.
2. Michie, D., and Chambers, R. A. (1968). BOXES: An experiment in adaptive control. *Machine Intelligence 2*, 137-152.
3. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
