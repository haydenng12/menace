from menace.game import board_from_key, board_key
from menace.players import MenacePlayer


def test_board_key_round_trip_for_inspector():
    key = "X-O-X----"
    assert board_key(board_from_key(key)) == key


def test_matchbox_probability_inputs_sum_correctly():
    menace = MenacePlayer(initial_beads=3)
    board = ["X", " ", "O", " ", "X", " ", " ", " ", " "]
    box = menace._ensure_matchbox(board)
    total = sum(box.values())
    probabilities = [beads / total for beads in box.values()]
    assert abs(sum(probabilities) - 1.0) < 1e-12
