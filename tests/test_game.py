from menace.game import X, make_move, new_board, valid_moves, winner


def test_new_board_has_nine_valid_moves():
    assert valid_moves(new_board()) == list(range(9))


def test_winner_detects_row():
    board = new_board()
    for move in (0, 1, 2):
        board = make_move(board, move, X)
    assert winner(board) == X
