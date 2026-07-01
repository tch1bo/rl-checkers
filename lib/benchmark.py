from draughts import Benchmark
from pydantic import BaseModel, Field

from lib.log import get_logger
from lib.utils import (
    DEFAULT_BOARD,
    BoardClassLiteral,
    EngineIdAndDepth,
    choose_board_class,
    make_engine,
)

logger = get_logger()


class BenchmarkArgs(BaseModel):
    n_games: int = Field(ge=1, default=10)
    a: EngineIdAndDepth
    b: EngineIdAndDepth = EngineIdAndDepth(engine="alpha-beta")
    board_type: BoardClassLiteral = DEFAULT_BOARD
    n_workers: int = Field(ge=1, default=10)

    def cli_cmd(self) -> None:
        benchmark(self)


def benchmark(args: BenchmarkArgs) -> None:
    logger.info("starting benchmarking", a=args.a, b=args.b, n_games=args.n_games)
    a = make_engine(args.a, args.board_type)
    b = make_engine(args.b, args.board_type)
    stats = Benchmark(
        a,
        b,
        board_class=choose_board_class(args.board_type),
        games=args.n_games,
        workers=args.n_workers,
        swap_colors=True,
    ).run()
    print(stats)
