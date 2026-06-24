from pathlib import Path
from typing import Any, Literal, Type, assert_never

from draughts import Engine

from lib.models import MLPQNet

EngineIdT = Literal["random", "alpha-beta"] | Path
BoardClassLiteral = Literal["russian", "standard"]
DEFAULT_BOARD: BoardClassLiteral = "standard"

import numpy as np
from draughts import (
    AlphaBetaEngine,
    BaseBoard,
    Engine,
    Move,
    RussianBoard,
    StandardBoard,
)


class RandomEngine(Engine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed=42)

    def get_best_move(
        self, board: BaseBoard, with_evaluation: bool = False
    ) -> Move | tuple[Move, float]:
        index = self._rng.integers(len(board.legal_moves))
        move = board.legal_moves[index]
        return (move, 1.0) if with_evaluation else move


def make_engine(engine_id: EngineIdT, board_type: BoardClassLiteral) -> Engine:
    if engine_id == "alpha-beta":
        return AlphaBetaEngine()
    if engine_id == "random":
        return RandomEngine()
    assert isinstance(engine_id, Path)

    mlp = MLPQNet.load_from_path(engine_id, choose_board_class(board_type))
    return mlp.as_engine()


def choose_board_class(b: BoardClassLiteral) -> Type[BaseBoard]:
    match b:
        case "russian":
            return RussianBoard
        case "standard":
            return StandardBoard
    assert_never(b)
