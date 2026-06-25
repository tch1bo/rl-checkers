from pathlib import Path
from typing import Any, Literal, Type, assert_never

import torch
from draughts import STANDARD_OPENINGS, AgentEngine, Benchmark, Engine
from draughts.benchmark import BenchmarkStats

from lib.models import MLPVNet

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

    # TODO(chibo): decide whether we need QNets at all. If yes, discriminate here.
    return MLPVNet.load_from_path(engine_id, choose_board_class(board_type)).as_engine()


def choose_board_class(b: BoardClassLiteral) -> Type[BaseBoard]:
    match b:
        case "russian":
            return RussianBoard
        case "standard":
            return StandardBoard
    assert_never(b)


def _num_games(board_class: Type[BaseBoard]) -> int:
    if board_class.SQUARES_COUNT == 50:
        return len(STANDARD_OPENINGS) * 2
    return 2


def benchmark_against_ab_engine(
    engine: AgentEngine, board_class: Type[BaseBoard]
) -> dict[int, BenchmarkStats]:
    num_games = _num_games(board_class)
    with torch.no_grad():
        result = {
            level: Benchmark(
                engine,
                AlphaBetaEngine(depth_limit=level),
                board_class=board_class,
                games=num_games,
                workers=1 if level < 5 else min(num_games, 10),
                swap_colors=True,
            ).run()
            for level in range(2, 8)
        }
    return result


def benchmark_against_random(
    engine: AgentEngine, board_class: Type[BaseBoard]
) -> BenchmarkStats:
    with torch.no_grad():
        return Benchmark(
            engine,
            RandomEngine(),
            board_class=board_class,
            games=_num_games(board_class),
            workers=1,
        ).run()


_SQ_CACHE: dict[int, np.ndarray] = {}


def _square_bits(squares_count: int) -> np.ndarray:
    sq = _SQ_CACHE.get(squares_count)
    if sq is None:
        sq = np.arange(squares_count, dtype=np.uint64)
        _SQ_CACHE[squares_count] = sq
    return sq


def encode_batch(
    masks: np.ndarray, perspectives: np.ndarray, squares_count: int
) -> np.ndarray:
    """Vectorized board-to-tensor encoding for a whole batch of states.

    Bit-identical to ``BaseBoard.to_tensor`` but decodes every state in one set of
    numpy ops instead of a per-square Python loop, so the fixed numpy overhead is
    amortized across the batch.

    Args:
        masks: ``(N, 4)`` uint64 array of raw bitboards, ordered
            ``[white_men, white_kings, black_men, black_kings]``.
        perspectives: ``(N,)`` array; ``+1`` = white to move, ``-1`` = black to move.
            Equivalent to ``board.turn`` at the moment the state was recorded.
        squares_count: number of squares on the board.

    Returns:
        ``(N, 4, squares_count)`` float32 array with channels
        ``[own_men, own_kings, opp_men, opp_kings]`` from each state's perspective.
    """
    sq = _square_bits(squares_count)
    bits = ((masks[:, :, None] >> sq) & np.uint64(1)).astype(np.float32)  # (N, 4, SC)
    wm, wk, bm, bk = bits[:, 0], bits[:, 1], bits[:, 2], bits[:, 3]

    white = (perspectives == 1)[:, None]  # (N, 1)
    out = np.empty_like(bits)
    out[:, 0] = np.where(white, wm, bm)  # own men
    out[:, 1] = np.where(white, wk, bk)  # own kings
    out[:, 2] = np.where(white, bm, wm)  # opp men
    out[:, 3] = np.where(white, bk, wk)  # opp kings
    return out
