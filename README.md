# MENACE Tic-Tac-Toe AI

A from-scratch Python implementation of **MENACE** (**M**atchbox **E**ducable **N**oughts **A**nd **C**rosses **E**ngine), the classic tic-tac-toe learning system created by Donald Michie in the 1960s.

Instead of using neural networks or minimax, MENACE learns using virtual **matchboxes and beads**. Every board state has a matchbox, and every legal move has a number of beads. MENACE randomly selects a move according to the bead counts and then changes those counts based on whether it won, drew, or lost.

## How it works

1. MENACE sees a tic-tac-toe board state.
2. That board state is used as the key for a virtual matchbox.
3. Every legal move starts with the same number of beads.
4. MENACE chooses a move randomly, weighted by bead count.
5. At the end of the game:
   - win: add beads to the moves it used
   - draw: add a smaller reward
   - loss: remove beads
6. Over many games, strong moves become more likely and weak moves become less likely.

This is a simple historical example of **reinforcement learning**.

## Project structure

```text
menace-github/
├── menace/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── game.py
│   ├── players.py
│   └── training.py
├── tests/
│   ├── test_game.py
│   └── test_menace.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements-dev.txt
└── README.md
```

## Run it

Requires Python 3.10+.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/menace-tictactoe.git
cd menace-tictactoe
```

### 2. Optional: create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
pip install -e .
```

## Train MENACE

Train against a random player for 50,000 games:

```bash
python -m menace train --games 50000
```

or, after installing the package:

```bash
menace train --games 50000
```

The learned matchboxes are saved to:

```text
menace_model.json
```

Train for more games:

```bash
python -m menace train --games 250000 --report-every 10000
```


## Visual training mode

Prefer something more interactive than terminal output? Launch the built-in Tkinter training dashboard:

```bash
python -m menace visual
```

The dashboard lets you choose the number of games, update interval, initial bead count, and output model. While training runs, it shows:

- live win, draw, and loss rates
- number of matchboxes MENACE has created
- training progress
- a learning-curve graph that updates as MENACE plays
- Start and Stop controls

The GUI uses Python's built-in `tkinter`, so no plotting package is required on standard Python installations.




## Board symmetry reduction

MENACE now recognizes rotations and reflections of the same tic-tac-toe
position as a single matchbox.

For example, these positions are treated as equivalent:

```text
X . .        . . X        . . .
. O .        . O .        . O .
. . .        . . .        . . X
```

Internally, every board is transformed into one canonical orientation before
MENACE looks up its matchbox. The chosen move is then transformed back to the
orientation of the real board.

This means:

- rotated positions share learning
- reflected positions share learning
- MENACE needs far fewer matchboxes
- experience generalizes across symmetric positions
- saved models are smaller and easier to inspect

Older saved model files are still supported: when loaded, equivalent old
matchboxes are automatically merged into canonical symmetry classes.


## Visual matchbox / bead inspector

The Tkinter dashboard now includes an **Inspect matchboxes** button.

Launch the GUI:

```bash
python -m menace visual
```

Train MENACE, then click **Inspect matchboxes**. The inspector lets you browse every learned board state and shows:

- the exact 3×3 board represented by the matchbox
- every legal move MENACE can make
- the bead count for each move
- a visual bar comparing bead counts
- the exact probability that MENACE will select each move
- total beads in the current matchbox
- previous/next controls for browsing learned states

If you already have a saved `menace_model.json`, the inspector can load it even before starting a new training session.


## Three extra experiments

### 1. Configurable reinforcement rules

You can change how strongly MENACE rewards or punishes moves:

```bash
python -m menace train --games 50000 --win-reward 5 --draw-reward 2 --loss-penalty 2
```

Defaults are:

```text
win reward:   +3 beads
draw reward:  +1 bead
loss penalty: -1 bead
```

This makes it easy to experiment with different learning behavior without changing the source code.

### 2. MENACE vs MENACE self-play

Train two independent MENACE players against each other:

```bash
python -m menace self-play --games 50000
```

This creates:

```text
menace_x.json
menace_o.json
```

Each player learns from the same games, receiving opposite rewards after wins and losses.

### 3. Compare MENACE against minimax

After training a normal model:

```bash
python -m menace train --games 50000
```

evaluate it against a perfect tic-tac-toe minimax player:

```bash
python -m menace compare --games 1000
```

The comparison tests MENACE both as X and as O. Minimax does not learn and never intentionally makes a suboptimal move, so this gives MENACE a much harder benchmark than a random opponent.


## Play against MENACE

Let MENACE move first:

```bash
python -m menace play
```

Play first as X:

```bash
python -m menace play --first
```

Allow MENACE to learn from your game and save the result:

```bash
python -m menace play --first --learn
```

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Example algorithm

A matchbox might look conceptually like this:

```python
{
    0: 3,
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 3,
}
```

The keys are possible moves and the values are bead counts. If move `4` repeatedly produces wins, its bead count increases, so MENACE becomes increasingly likely to select the center square in that state.

## What makes this interesting?

MENACE was originally implemented physically using hundreds of matchboxes filled with colored beads. It demonstrates that a machine can improve its behavior through rewards and penalties without being explicitly told the best move for every situation.

This implementation keeps the original idea but represents the matchboxes using Python dictionaries and saves learned states as JSON.