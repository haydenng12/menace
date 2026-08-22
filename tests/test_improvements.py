
from menace.game import X, O, new_board
from menace.players import MenacePlayer, MinimaxPlayer
from menace.training import evaluate_against_minimax, train_self_play


def test_minimax_has_valid_opening_move():
    move = MinimaxPlayer().get_move(new_board(), X)
    assert move in range(9)


def test_self_play_runs():
    x = MenacePlayer()
    o = MenacePlayer()
    stats = train_self_play(x, o, games=20, report_every=0)
    assert stats.games == 20
    assert stats.wins + stats.draws + stats.losses == 20


def test_minimax_evaluation_runs():
    menace = MenacePlayer()
    stats = evaluate_against_minimax(menace, games=10, menace_symbol=X)
    assert stats.games == 10
    assert stats.wins + stats.draws + stats.losses == 10
