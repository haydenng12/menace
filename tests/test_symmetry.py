
from menace.game import X, O, new_board
from menace.players import MenacePlayer
from menace.symmetry import TRANSFORMS, canonicalize, transform_board


def test_all_eight_symmetries_share_one_canonical_key():
    board = [
        X, " ", O,
        " ", X, " ",
        " ", " ", " ",
    ]
    keys = {
        canonicalize(transform_board(board, transform)).key
        for transform in TRANSFORMS
    }
    assert len(keys) == 1


def test_symmetric_boards_share_one_matchbox():
    player = MenacePlayer()
    board = [
        X, " ", O,
        " ", X, " ",
        " ", " ", " ",
    ]

    rotated = transform_board(board, TRANSFORMS[1])
    player._ensure_matchbox(board)
    player._ensure_matchbox(rotated)

    assert len(player.matchboxes) == 1


def test_move_mapping_returns_legal_move_on_rotated_board():
    player = MenacePlayer()
    board = [
        X, " ", O,
        " ", X, " ",
        " ", " ", " ",
    ]

    for transform in TRANSFORMS:
        symmetric = transform_board(board, transform)
        move = player.get_move(symmetric, O)
        assert symmetric[move] == " "
        player.reset_history()


def test_empty_board_always_uses_one_matchbox():
    player = MenacePlayer()
    player.get_move(new_board(), X)
    assert len(player.matchboxes) == 1
