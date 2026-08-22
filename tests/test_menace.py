from menace.game import new_board
from menace.players import MenacePlayer


def test_menace_creates_matchbox_and_returns_legal_move():
    player = MenacePlayer(initial_beads=3)
    board = new_board()
    move = player.get_move(board, "X")
    assert 0 <= move <= 8
    assert len(player.matchboxes) == 1


def test_learning_rewards_selected_move():
    player = MenacePlayer(initial_beads=3)
    board = new_board()
    move = player.get_move(board, "X")
    key = next(iter(player.matchboxes))
    before = player.matchboxes[key][move]
    player.learn("win")
    assert player.matchboxes[key][move] == before + 3
