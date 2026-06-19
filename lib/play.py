from draughts import Server
from pydantic import BaseModel

from lib.utils import (
    DEFAULT_BOARD,
    BoardClassLiteral,
    EngineIdT,
    choose_board_class,
    make_engine,
)


class PlayArgs(BaseModel):
    white: EngineIdT
    black: EngineIdT = "alpha-beta"
    board_type: BoardClassLiteral = DEFAULT_BOARD

    def cli_cmd(self) -> None:
        play(self)


def play(args: PlayArgs) -> None:
    board_class = choose_board_class(args.board_type)
    server = Server(
        board=board_class(),
        white_engine=make_engine(args.white, args.board_type),
        black_engine=make_engine(args.black, args.board_type),
    )
    server.run()
