from menace.cli import build_parser


def test_visual_command_exists():
    args = build_parser().parse_args(["visual"])
    assert args.command == "visual"
    assert callable(args.func)
