"""Tkinter visualizer for MENACE training."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from .game import EMPTY, X, board_from_key
from .players import MenacePlayer
from .training import train_against_random

DEFAULT_MODEL = Path("menace_model.json")


class TrainingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MENACE Training Lab")
        self.root.geometry("980x680")
        self.root.minsize(820, 600)

        self.messages: queue.Queue = queue.Queue()
        self.running = False
        self.stop_requested = False
        self.menace: MenacePlayer | None = None
        self.history: list[tuple[int, float, float, float]] = []

        self.games_var = tk.StringVar(value="50000")
        self.batch_var = tk.StringVar(value="500")
        self.beads_var = tk.StringVar(value="3")
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        self._build()
        self.root.after(80, self._poll)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="MENACE Training Lab", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Watch the matchbox learner improve against a random tic-tac-toe player.",
        ).pack(anchor="w", pady=(2, 16))

        controls = ttk.LabelFrame(outer, text="Training setup", padding=12)
        controls.pack(fill="x")

        fields = [
            ("Games", self.games_var),
            ("Update every", self.batch_var),
            ("Initial beads", self.beads_var),
            ("Model file", self.model_var),
        ]
        for col, (label, variable) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=col, sticky="w", padx=(0, 8))
            ttk.Entry(controls, textvariable=variable, width=18).grid(
                row=1, column=col, sticky="ew", padx=(0, 12)
            )
            controls.columnconfigure(col, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=12)
        self.start_btn = ttk.Button(buttons, text="Start training", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        ttk.Button(buttons, text="Clear graph", command=self.clear_graph).pack(side="left")
        ttk.Button(buttons, text="Inspect matchboxes", command=self.open_inspector).pack(side="left", padx=8)

        ttk.Progressbar(outer, variable=self.progress_var, maximum=100).pack(fill="x")
        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(5, 12))

        cards = ttk.Frame(outer)
        cards.pack(fill="x")
        self.win_label = self._card(cards, "Win rate", "0.0%", 0)
        self.draw_label = self._card(cards, "Draw rate", "0.0%", 1)
        self.loss_label = self._card(cards, "Loss rate", "0.0%", 2)
        self.box_label = self._card(cards, "Matchboxes", "0", 3)

        graph_frame = ttk.LabelFrame(outer, text="Learning curve", padding=8)
        graph_frame.pack(fill="both", expand=True, pady=(14, 0))
        self.canvas = tk.Canvas(graph_frame, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.draw_graph())

        legend = ttk.Label(
            outer,
            text="Graph lines: W = win rate   D = draw rate   L = loss rate",
        )
        legend.pack(anchor="e", pady=(5, 0))

    def _card(self, parent: ttk.Frame, title: str, value: str, col: int) -> ttk.Label:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
        parent.columnconfigure(col, weight=1)
        label = ttk.Label(frame, text=value, font=("Segoe UI", 18, "bold"))
        label.pack()
        return label

    def start(self) -> None:
        if self.running:
            return
        try:
            games = int(self.games_var.get())
            batch = int(self.batch_var.get())
            beads = int(self.beads_var.get())
            if games < 1 or batch < 1 or beads < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid settings", "Games, update interval, and beads must be positive integers.")
            return

        self.running = True
        self.stop_requested = False
        self.history.clear()
        self.menace = MenacePlayer(initial_beads=beads)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Training...")
        self.progress_var.set(0)
        self.draw_graph()

        thread = threading.Thread(
            target=self._train_worker,
            args=(games, batch, self.model_var.get()),
            daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        self.stop_requested = True
        self.status_var.set("Stopping after the current batch...")

    def clear_graph(self) -> None:
        if not self.running:
            self.history.clear()
            self.draw_graph()

    def _train_worker(self, games: int, batch: int, model: str) -> None:
        assert self.menace is not None
        completed = wins = draws = losses = 0

        while completed < games and not self.stop_requested:
            amount = min(batch, games - completed)
            stats = train_against_random(
                self.menace,
                games=amount,
                menace_symbol=X,
                report_every=0,
            )
            completed += stats.games
            wins += stats.wins
            draws += stats.draws
            losses += stats.losses
            self.messages.put(("progress", completed, games, wins, draws, losses, len(self.menace.matchboxes)))

        self.menace.save(model)
        self.messages.put(("done", completed, games, model, self.stop_requested))

    def _poll(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                if message[0] == "progress":
                    _, done, total, wins, draws, losses, boxes = message
                    wr = wins / done
                    dr = draws / done
                    lr = losses / done
                    self.history.append((done, wr, dr, lr))
                    self.progress_var.set(done / total * 100)
                    self.status_var.set(f"{done:,} / {total:,} games completed")
                    self.win_label.config(text=f"{wr:.1%}")
                    self.draw_label.config(text=f"{dr:.1%}")
                    self.loss_label.config(text=f"{lr:.1%}")
                    self.box_label.config(text=f"{boxes:,}")
                    self.draw_graph()
                elif message[0] == "done":
                    _, done, total, model, stopped = message
                    self.running = False
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status_var.set(
                        f"{'Stopped' if stopped else 'Finished'} at {done:,} games. Saved to {model}"
                    )
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _load_model_for_inspector(self) -> MenacePlayer | None:
        """Return the current learner or load the configured model from disk."""
        if self.menace is not None and self.menace.matchboxes:
            return self.menace

        model_path = Path(self.model_var.get())
        if not model_path.exists():
            messagebox.showinfo(
                "No learned matchboxes yet",
                "Train MENACE first, or set Model file to an existing MENACE model.",
            )
            return None

        try:
            self.menace = MenacePlayer.load(model_path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Could not load model", str(exc))
            return None

        self.box_label.config(text=f"{len(self.menace.matchboxes):,}")
        return self.menace

    def open_inspector(self) -> None:
        menace = self._load_model_for_inspector()
        if menace is None:
            return
        MatchboxInspector(self.root, menace)

    def draw_graph(self) -> None:
        c = self.canvas
        c.delete("all")
        width = max(c.winfo_width(), 200)
        height = max(c.winfo_height(), 180)
        left, right, top, bottom = 55, 18, 20, 38
        gw = width - left - right
        gh = height - top - bottom

        # Axes and horizontal grid.
        c.create_line(left, top, left, top + gh, fill="#555")
        c.create_line(left, top + gh, left + gw, top + gh, fill="#555")
        for pct in (0, 25, 50, 75, 100):
            y = top + gh * (1 - pct / 100)
            c.create_line(left, y, left + gw, y, fill="#e8e8e8")
            c.create_text(left - 8, y, text=f"{pct}%", anchor="e", fill="#555")

        if not self.history:
            c.create_text(width / 2, height / 2, text="Start training to see MENACE learn.", fill="#777")
            return

        max_games = max(point[0] for point in self.history)
        c.create_text(left, top + gh + 22, text="0", anchor="w", fill="#555")
        c.create_text(left + gw, top + gh + 22, text=f"{max_games:,} games", anchor="e", fill="#555")

        # Tkinter colors are used only inside the app itself to distinguish series.
        series = [(1, "#2e8b57", "W"), (2, "#c28b00", "D"), (3, "#b22222", "L")]
        for index, color, tag in series:
            coords = []
            for point in self.history:
                x = left + (point[0] / max_games) * gw
                y = top + (1 - point[index]) * gh
                coords.extend((x, y))
            if len(coords) >= 4:
                c.create_line(*coords, fill=color, width=2, smooth=True)
            else:
                c.create_oval(coords[0]-2, coords[1]-2, coords[0]+2, coords[1]+2, fill=color, outline=color)
            x, y = coords[-2], coords[-1]
            c.create_text(x - 6, y - 8, text=tag, fill=color, anchor="e")



class MatchboxInspector:
    """Interactive browser for MENACE's learned virtual matchboxes."""

    CELL_SIZE = 88

    def __init__(self, parent: tk.Tk, menace: MenacePlayer) -> None:
        self.menace = menace
        self.keys = sorted(menace.matchboxes.keys(), key=lambda key: (key.count("-"), key))
        self.current_key = tk.StringVar()

        self.window = tk.Toplevel(parent)
        self.window.title("MENACE Matchbox Inspector")
        self.window.geometry("980x650")
        self.window.minsize(840, 560)

        self._build()
        self.refresh_states()

    def _build(self) -> None:
        outer = ttk.Frame(self.window, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Visual Matchbox / Bead Inspector",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Each board state is a virtual matchbox. More beads mean MENACE "
                "is more likely to choose that move."
            ),
        ).pack(anchor="w", pady=(2, 12))

        selector = ttk.Frame(outer)
        selector.pack(fill="x")
        ttk.Label(selector, text="Board state:").pack(side="left")
        self.combo = ttk.Combobox(
            selector,
            textvariable=self.current_key,
            state="readonly",
            width=24,
        )
        self.combo.pack(side="left", padx=8)
        self.combo.bind("<<ComboboxSelected>>", lambda _event: self.render_current())

        ttk.Button(selector, text="Previous", command=lambda: self.shift_state(-1)).pack(side="left")
        ttk.Button(selector, text="Next", command=lambda: self.shift_state(1)).pack(side="left", padx=6)
        ttk.Button(selector, text="Refresh", command=self.refresh_states).pack(side="left")

        self.summary = ttk.Label(selector, text="")
        self.summary.pack(side="right")

        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True, pady=(14, 0))
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        board_frame = ttk.LabelFrame(content, text="Board state", padding=14)
        board_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 14))

        self.board_canvas = tk.Canvas(
            board_frame,
            width=self.CELL_SIZE * 3,
            height=self.CELL_SIZE * 3,
            background="white",
            highlightthickness=0,
        )
        self.board_canvas.pack()

        self.board_key_label = ttk.Label(board_frame, text="", font=("Consolas", 10))
        self.board_key_label.pack(pady=(10, 0))

        bars_frame = ttk.LabelFrame(content, text="Move beads and probabilities", padding=14)
        bars_frame.grid(row=0, column=1, sticky="nsew")
        bars_frame.columnconfigure(1, weight=1)

        ttk.Label(bars_frame, text="Move", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(bars_frame, text="Relative bead count", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=8
        )
        ttk.Label(bars_frame, text="Beads", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=8)
        ttk.Label(bars_frame, text="Probability", font=("Segoe UI", 10, "bold")).grid(row=0, column=3, padx=8)

        self.bar_area = ttk.Frame(bars_frame)
        self.bar_area.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
        self.bar_area.columnconfigure(1, weight=1)

        explanation = ttk.LabelFrame(outer, text="How to read this", padding=10)
        explanation.pack(fill="x", pady=(14, 0))
        ttk.Label(
            explanation,
            text=(
                "MENACE draws a virtual bead at random. A move with 20 beads is twice as likely "
                "to be selected as a move with 10 beads. Training rewards successful moves by "
                "adding beads and penalizes losing moves by removing beads."
            ),
            wraplength=900,
            justify="left",
        ).pack(anchor="w")

    def refresh_states(self) -> None:
        self.keys = sorted(
            self.menace.matchboxes.keys(),
            key=lambda key: (-key.count("-"), key),
        )
        self.combo["values"] = self.keys
        self.summary.config(text=f"{len(self.keys):,} learned matchboxes")

        if not self.keys:
            self.current_key.set("")
            self.render_current()
            return

        if self.current_key.get() not in self.keys:
            self.current_key.set(self.keys[0])
        self.render_current()

    def shift_state(self, direction: int) -> None:
        if not self.keys:
            return
        try:
            index = self.keys.index(self.current_key.get())
        except ValueError:
            index = 0
        index = (index + direction) % len(self.keys)
        self.current_key.set(self.keys[index])
        self.combo.current(index)
        self.render_current()

    def render_current(self) -> None:
        key = self.current_key.get()
        if not key or key not in self.menace.matchboxes:
            self.board_canvas.delete("all")
            for child in self.bar_area.winfo_children():
                child.destroy()
            self.board_key_label.config(text="No learned matchboxes available.")
            return

        board = board_from_key(key)
        box = self.menace.matchboxes[key]
        self._draw_board(board, box)
        self._draw_beads(box)
        total = sum(box.values())
        self.board_key_label.config(
            text=f"Key: {key}    Total beads: {total:,}"
        )

    def _draw_board(self, board: list[str], box: dict[int, int]) -> None:
        c = self.board_canvas
        c.delete("all")
        size = self.CELL_SIZE

        for index in range(9):
            row, col = divmod(index, 3)
            x1, y1 = col * size, row * size
            x2, y2 = x1 + size, y1 + size
            c.create_rectangle(x1, y1, x2, y2, outline="#444", width=2)

            value = board[index]
            if value != EMPTY:
                c.create_text(
                    x1 + size / 2,
                    y1 + size / 2,
                    text=value,
                    font=("Segoe UI", 30, "bold"),
                )
            elif index in box:
                c.create_text(
                    x1 + size / 2,
                    y1 + size / 2 - 8,
                    text=str(index + 1),
                    font=("Segoe UI", 16, "bold"),
                    fill="#666",
                )
                c.create_text(
                    x1 + size / 2,
                    y1 + size / 2 + 16,
                    text=f"{box[index]} beads",
                    font=("Segoe UI", 9),
                    fill="#777",
                )

    def _draw_beads(self, box: dict[int, int]) -> None:
        for child in self.bar_area.winfo_children():
            child.destroy()

        if not box:
            ttk.Label(self.bar_area, text="No legal moves in this matchbox.").grid(
                row=0, column=0, sticky="w"
            )
            return

        total = sum(box.values())
        maximum = max(box.values())

        for row, (move, beads) in enumerate(sorted(box.items())):
            probability = beads / total if total else 0.0

            ttk.Label(self.bar_area, text=f"Square {move + 1}").grid(
                row=row, column=0, sticky="w", pady=5
            )

            bar = tk.Canvas(
                self.bar_area,
                height=24,
                background="white",
                highlightthickness=1,
                highlightbackground="#ddd",
            )
            bar.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            bar.bind(
                "<Configure>",
                lambda event, widget=bar, value=beads, max_value=maximum:
                    self._paint_bar(widget, value, max_value),
            )

            ttk.Label(self.bar_area, text=f"{beads:,}").grid(
                row=row, column=2, padx=8
            )
            ttk.Label(self.bar_area, text=f"{probability:.1%}").grid(
                row=row, column=3, padx=8
            )

    @staticmethod
    def _paint_bar(canvas: tk.Canvas, value: int, maximum: int) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        fraction = value / maximum if maximum else 0
        canvas.create_rectangle(
            0,
            0,
            width * fraction,
            height,
            fill="#4f81bd",
            outline="",
        )

def launch_visualizer() -> None:
    root = tk.Tk()
    TrainingApp(root)
    root.mainloop()
