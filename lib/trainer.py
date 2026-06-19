import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Type

import numpy as np
import torch
import tqdm
from draughts import AlphaBetaEngine, BaseBoard, Benchmark, BenchmarkStats, Color
from draughts import Move as DraughtsMove
from pydantic import BaseModel, Field, PrivateAttr
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from lib.log import get_logger
from lib.mlp import MLPQNet
from lib.utils import DEFAULT_BOARD, BoardClassLiteral, RandomEngine, choose_board_class

logger = get_logger()


class TrainArgs(BaseModel):
    board_type: BoardClassLiteral = DEFAULT_BOARD

    learning_rate: float = Field(ge=0.0, default=1e-4)
    gamma: float = Field(ge=0.0, le=1.0, default=0.99)

    max_moves_per_game: int = Field(ge=0, default=1000)
    min_replay_buffer: int = Field(
        ge=0,
        default=1000,
        description="only start training when the replay buffer has that many samples",
    )
    games_per_gradient_step: int = Field(ge=1, default=1)
    max_replay_buffer: int = Field(ge=0, default=100_000)
    eps_min: float = Field(ge=0, le=1.0, default=0.1)
    eps_decay_ratio: float = Field(
        ge=0,
        le=1.0,
        default=0.3,
        description="eps will be decayed from 1.0 to its min over this ratio of epochs",
    )
    num_gradient_updates: int = Field(ge=0, default=100000)
    train_batch_size: int = Field(ge=0, default=1000)

    inference_batch_size: int = Field(ge=0, default=1000)

    seed: int = 42
    sync_every: int = Field(
        default=1000,
        description="number of steps between online->target network weight syncs",
    )
    steps_in_epoch: int = Field(
        default=10000,
        description="number of steps between saves (and benchmarking runs)",
    )
    _rng: np.random.Generator | None = PrivateAttr(default=None)
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"/tmp/checkers/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )

    def cli_cmd(self) -> None:
        train(self)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng


class Move(BaseModel):
    pre_fen: str
    post_fen: str
    action_uci: str

    # `reward` is:
    #   1 - if the move was the last move of the winning side
    #  -1 - if the move was the last move of the losing side
    #   0 - otherwise
    reward: int
    is_final: bool
    color: Color


def eps_greedy_move(
    args: TrainArgs, eps: float, qnet: MLPQNet, board: BaseBoard
) -> DraughtsMove:
    if args.rng.random() > eps:
        # Greedy move
        return qnet.select_move(board)

    # Exploration move
    moves = board.legal_moves
    return moves[args.rng.integers(len(moves))]


def play_one_game(
    args: TrainArgs,
    qnet: MLPQNet,
    board_class: Type[BaseBoard],
    replay_buffer: list[Move],
    eps: float,
) -> None:
    board = board_class()
    fens: list[str] = [board.fen]
    ucis: list[str] = []
    while not board.game_over:
        move = eps_greedy_move(args, eps, qnet, board)
        board.push(move)
        fens.append(board.fen)
        ucis.append(str(move))

    # Transforms the fens and ucis into `Moves`
    moves: list[Move] = []
    for i, uci in enumerate(ucis):
        moves.append(
            Move(
                pre_fen=fens[i],
                post_fen=fens[min(i + 2, len(fens) - 1)],
                action_uci=uci,
                reward=0,
                # Only the last two moves are final
                is_final=board.game_over and i >= len(ucis) - 2,
                color=Color.BLACK if i % 2 else Color.WHITE,
            )
        )
    if board.game_over and not board.is_draw:
        # The last move of the losing side
        moves[-2].reward = -1
        # The last move of the winning side
        moves[-1].reward = 1
    replay_buffer.extend(moves)


FenAndColor = tuple[str, Color]


def optimize(
    args: TrainArgs,
    online_model: MLPQNet,
    target_model: MLPQNet,
    optimizer: torch.optim.Optimizer,
    board_class: Type[BaseBoard],
    replay_buffer: list[Move],
) -> float:
    indices = args.rng.choice(np.arange(len(replay_buffer)), args.train_batch_size)
    batch = [replay_buffer[idx] for idx in indices]

    # Compute the bootstrap term `max Q(S', a)`
    target_model.eval()
    with torch.no_grad():
        post_inputs = torch.empty((len(batch), 4, board_class.SQUARES_COUNT))
        mask = torch.empty((len(batch), board_class.SQUARES_COUNT**2), dtype=torch.bool)
        for i, move in enumerate(batch):
            board = board_class.from_fen(move.post_fen)
            post_inputs[i] = Tensor(board.to_tensor(perspective=move.color))
            mask[i] = Tensor(board.legal_moves_mask())

        post_inputs = post_inputs.to(device=target_model.device)
        mask = mask.to(device=target_model.device)

        values = target_model.forward(post_inputs)
        values.masked_fill_(~mask, float("-inf"))
        max_values = values.amax(dim=1)

        is_final_move = torch.tensor(
            [move.is_final for move in batch],
            dtype=torch.bool,
            device=target_model.device,
        )
        max_values.masked_fill_(is_final_move, 0.0)
        rewards = torch.tensor(
            [move.reward for move in batch], device=online_model.device
        )

        targets = args.gamma * max_values + rewards

    # Do batched SGD
    online_model.train()
    pre_inputs = torch.empty((len(batch), 4, board_class.SQUARES_COUNT))
    move_indices = torch.zeros(len(batch), dtype=torch.int)
    for j, move in enumerate(batch):
        board = board_class.from_fen(move.pre_fen)
        pre_inputs[j] = Tensor(board.to_tensor(perspective=move.color))
        m = DraughtsMove.from_uci(move.action_uci, board.legal_moves)
        move_indices[j] = board.move_to_index(m)

    pre_inputs = pre_inputs.to(online_model.device)
    move_indices = move_indices.to(online_model.device)

    y = online_model.forward(pre_inputs)
    predictions = y[torch.arange(y.size(0), device=online_model.device), move_indices]
    loss = mse_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def benchmark_against_ab_engine(
    qnet: MLPQNet, board_class: Type[BaseBoard]
) -> dict[int, BenchmarkStats]:
    qnet = deepcopy(qnet).to("cpu")
    result = {
        level: Benchmark(
            qnet.as_engine(),
            AlphaBetaEngine(depth_limit=level),
            board_class=board_class,
            games=10,
            workers=10,
        ).run()
        for level in range(2, 8)
    }
    return result


def benchmark_against_random(
    qnet: MLPQNet, board_class: Type[BaseBoard]
) -> BenchmarkStats:
    qnet = deepcopy(qnet).to("cpu")
    return Benchmark(
        qnet.as_engine(),
        RandomEngine(),
        board_class=board_class,
        games=10,
        workers=10,
    ).run()


def train(args: TrainArgs) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(args.model_dump_json(indent=2))
    tb_writer = SummaryWriter(log_dir=args.out_dir)

    board_class = choose_board_class(args.board_type)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logger.info("starting training", out_dir=args.out_dir, device=device)

    online_model = MLPQNet.init_with_random_weights(board_class, device)
    target_model = deepcopy(online_model)

    # Fill up the replay buffer with moves played by random policies
    replay_buffer: list[Move] = []
    while len(replay_buffer) < args.min_replay_buffer:
        play_one_game(args, online_model, board_class, replay_buffer, 1.0)

    # Do the "play -> gradient update" steps
    eps_decay_steps = max(int(args.eps_decay_ratio * args.num_gradient_updates), 1)
    optimizer = torch.optim.RMSprop(online_model.parameters(), lr=args.learning_rate)
    loss_sum: float = 0
    start_time = time.perf_counter()
    for step in tqdm.trange(args.num_gradient_updates):
        eps = max(args.eps_min, 1.0 - step / eps_decay_steps)

        # Play games
        for _ in range(args.games_per_gradient_step):
            play_one_game(args, online_model, board_class, replay_buffer, eps)

        # Clean the storage of the replay buffer if it gets too big
        if len(replay_buffer) > 10 * args.max_replay_buffer:
            replay_buffer = replay_buffer[-args.max_replay_buffer :]

        # Do a gradient step
        loss = optimize(
            args,
            online_model,
            target_model,
            optimizer,
            board_class,
            replay_buffer[-args.max_replay_buffer :],
        )
        loss_sum += loss
        tb_writer.add_scalar("loss", loss, step)

        if step % args.sync_every == 0:
            # sync online_model -> target_model
            target_model = deepcopy(online_model)

        if step % args.steps_in_epoch == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"checkpoint at {step} steps",
                mean_loss=f"{loss_sum / args.steps_in_epoch:.4f}",
                elapsed=f"{elapsed:.4f}s",
                elapsed_per_step=f"{elapsed/args.steps_in_epoch:.4f}s",
                eps=f"{eps:.3f}",
            )
            loss_sum = 0.0

            # Benchmark against an alpha-beta engine
            logger.info("running benchmarks against AB and random engines")
            vs_ab = benchmark_against_ab_engine(online_model, board_class)
            for level, stats in vs_ab.items():
                tb_writer.add_scalar(f"win-rate-ab-{level}", stats.e1_win_rate, step)
                tb_writer.add_scalar(f"game-len-ab-{level}", stats.avg_moves, step)

            # Benchmark against a random engine
            vs_random = benchmark_against_random(online_model, board_class)
            tb_writer.add_scalar(f"win-rate-random", vs_random.e1_win_rate, step)
            tb_writer.add_scalar(f"game-len-random", vs_random.avg_moves, step)
            tb_writer.flush()

            out_path = args.out_dir / f"checkpoint_{step}.pt"
            logger.info("saved checkpoint", out_path=out_path)
            torch.save(online_model.state_dict(), out_path)
            start_time = time.perf_counter()
