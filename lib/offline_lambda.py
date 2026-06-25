import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Type

import numpy as np
import torch
import tqdm
from draughts import BaseBoard, Color, Move
from pydantic import BaseModel, Field, PrivateAttr
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from lib.log import get_logger
from lib.models import NUM_CHANNELS, MLPVNet
from lib.utils import (
    DEFAULT_BOARD,
    BoardClassLiteral,
    benchmark_against_ab_engine,
    benchmark_against_random,
    choose_board_class,
)

logger = get_logger()


class TrainArgs(BaseModel):
    """This trainer implements an offline lambda-return algorithm"""

    board_type: BoardClassLiteral = DEFAULT_BOARD

    learning_rate: float = Field(ge=0.0, default=1e-4)
    gamma: float = Field(ge=0.0, le=1.0, default=1.0)
    lam: float = Field(ge=0.0, le=1.0, default=0.75)

    games_per_step: int = Field(lt=1, default=100)
    max_moves_per_game: int = Field(ge=0, default=200)
    eps_min: float = Field(ge=0, le=1.0, default=0.1)
    eps_decay_ratio: float = Field(
        ge=0,
        le=1.0,
        default=0.3,
        description="eps will be decayed from 1.0 to its min over this ratio of epochs",
    )
    num_steps: int = Field(ge=0, default=100000)
    train_batch_size: int = Field(ge=0, default=128)

    seed: int = 42
    steps_in_epoch: int = Field(
        default=10000,
        description="number of steps between saves (and benchmarking runs)",
    )
    _rng: np.random.Generator | None = PrivateAttr(default=None)
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"/tmp/checkers/offline_lambda_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )

    def cli_cmd(self) -> None:
        train(self)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng


def play_games_and_sample_batch(
    args: TrainArgs, model: MLPVNet, board_class: Type[BaseBoard], eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    states_and_values: list[list[tuple[torch.Tensor, float]]] = [
        [] for _ in range(args.games_per_step)
    ]
    boards: list[BaseBoard] = [board_class() for _ in range(args.games_per_step)]

    # Play N games using the current policy and record all the states and their values
    for _ in range(args.max_moves_per_game):
        # The indices of games that are not over yet
        active_game_indices: list[int] = []

        # For each game in `active_game_indices` the moves available from the current state
        # Together with each move, stores the index of the post-move state tensor in `state_tensors`
        moves_for_game: list[list[tuple[Move, int]]] = []

        # For each game in `active_game_indices` stores the index of the tensor for the current state
        # of the board
        pre_state_indices: list[int] = []

        # Stores all the states to evaluate by the model
        state_tensors: list[torch.Tensor] = []

        # Choose the games that are still going
        for game_idx in range(args.games_per_step):
            board = boards[game_idx]
            if board.game_over:
                continue

            active_game_indices.append(game_idx)
            pre_state_indices.append(len(state_tensors))

            # The pre-state is from the mover's perspective
            state_tensors.append(torch.from_numpy(board.to_tensor()))

            # TODO(chibo): this doesn't do any exploration :(

            moves_for_game.append([])
            for move in board.legal_moves:
                board.push(move)

                moves_for_game[-1].append((move, len(state_tensors)))
                # The post state is from the opponent's perspective
                state_tensors.append(torch.from_numpy(board.to_tensor()))
                board.pop()

        if not active_game_indices:
            break

        assert len(active_game_indices) == len(moves_for_game)
        assert len(active_game_indices) == len(pre_state_indices)
        num_moves_for_game = [len(moves) for moves in moves_for_game]

        # For every game, there should be:
        #   1. a tensor for the state before any moves
        #   2. a tensor for the state after each move
        assert sum(num_moves_for_game) + len(active_game_indices) == len(state_tensors)

        inputs = torch.stack(state_tensors).to(model.device)
        with torch.no_grad():
            values = model.forward(inputs).to("cpu")

        for i, game_idx in enumerate(active_game_indices):
            # Push the pre-state tensor and its value
            pre_state_idx = pre_state_indices[i]
            states_and_values[game_idx].append(
                (state_tensors[pre_state_idx], float(values[pre_state_idx].item()))
            )

            # Choose and make the best move
            moves_and_indices = moves_for_game[i]
            if args.rng.random() > eps:
                # Greedy move
                # Since the post state is from the opponent's perspective, we need to take `min`
                # (and not `max`) here
                best_move_idx = int(
                    values[[mi[1] for mi in moves_and_indices]].argmin().item()
                )
            else:
                # Exploration move
                best_move_idx = int(args.rng.integers(len(moves_and_indices)))
            boards[game_idx].push(moves_and_indices[best_move_idx][0])

    # Determine the winner
    winner: list[Color | None] = []
    for board in boards:
        if not board.game_over or board.is_draw:
            winner.append(None)
            continue

        winner.append(Color.WHITE if board.turn == Color.BLACK else Color.BLACK)

    # Sample a batch of states
    all_move_indices = [
        (game_idx, move_idx)
        for game_idx, svs in enumerate(states_and_values)
        for move_idx in range(len(svs))
    ]
    if len(all_move_indices) < args.train_batch_size:
        sampled_indices = all_move_indices
    else:
        idx = args.rng.choice(
            np.arange(len(all_move_indices)), args.train_batch_size, replace=False
        )
        sampled_indices = [all_move_indices[i] for i in idx]

    sampled_states = torch.empty(
        (len(sampled_indices), NUM_CHANNELS, board_class.SQUARES_COUNT)
    )
    sampled_updates = torch.zeros(len(sampled_indices))

    for i, (game_idx, move_idx) in enumerate(sampled_indices):
        svs = states_and_values[game_idx]

        # Compute the update for this state by iterating only over the states for this player
        update = 0.0
        lam = 1.0
        gamma = 1.0
        # `sign` is needed to negate the value of the opponent's state
        sign = -1.0
        for n in range(move_idx + 1, len(svs)):
            gamma *= args.gamma
            update += sign * lam * gamma * svs[n][1]
            lam *= args.lam
            sign *= -1

        update = (1 - args.lam) * update

        if winner[game_idx] is not None:
            reward = lam * gamma
            if winner[game_idx] == Color.WHITE and move_idx % 2 == 1:
                reward = -reward
            if winner[game_idx] == Color.BLACK and move_idx % 2 == 0:
                reward = -reward
            update += reward

        sampled_updates[i] = update
        sampled_states[i] = svs[move_idx][0]

    return sampled_states, sampled_updates


def optimize(
    model: MLPVNet,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    updates: torch.Tensor,
) -> float:
    model.train()
    states = states.to(model.device)
    updates = updates.to(model.device)

    y = model.forward(states).reshape(-1)
    loss = mse_loss(y, updates)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(args: TrainArgs) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(args.model_dump_json(indent=2))
    tb_writer = SummaryWriter(log_dir=args.out_dir)

    board_class = choose_board_class(args.board_type)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    logger.info("starting training", out_dir=args.out_dir, device=device)

    # TODO(chibo): sample the states for validation

    model = MLPVNet.init_with_random_weights(board_class, device)
    eps_decay_steps = max(int(args.eps_decay_ratio * args.num_steps), 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_time = time.perf_counter()
    play_time = 0.0
    opt_time = 0.0

    for step in tqdm.trange(args.num_steps):
        eps = max(args.eps_min, 1.0 - step / eps_decay_steps)

        # Play games and sample a batch of states and corresponding updates
        play_start = time.perf_counter()
        states, updates = play_games_and_sample_batch(args, model, board_class, eps)
        play_time += time.perf_counter() - play_start

        # Optimize the model on the sampled batch
        opt_start = time.perf_counter()
        loss = optimize(model, optimizer, states, updates)
        opt_time += time.perf_counter() - opt_start
        tb_writer.add_scalar("loss", loss, step)

        if step % args.steps_in_epoch == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"checkpoint at {step} steps",
                elapsed=f"{elapsed:.4f}s",
                elapsed_per_step=f"{elapsed/args.steps_in_epoch:.4f}s",
                eps=f"{eps:.2f}",
                play_time_per_step=f"{play_time/args.steps_in_epoch:.4f}s",
                opt_time_per_step=f"{opt_time/args.steps_in_epoch:.4f}s",
            )

            # Benchmark against an alpha-beta engine
            logger.info("running benchmarks against AB and random engines")
            cpu_online_model = deepcopy(model).to("cpu").eval()
            vs_ab = benchmark_against_ab_engine(
                cpu_online_model.as_engine(), board_class
            )
            for level, stats in vs_ab.items():
                tb_writer.add_scalar(f"win-rate-ab-{level}", stats.e1_win_rate, step)
                tb_writer.add_scalar(f"game-len-ab-{level}", stats.avg_moves, step)

            # Benchmark against a random engine
            vs_random = benchmark_against_random(
                cpu_online_model.as_engine(), board_class
            )
            tb_writer.add_scalar(f"win-rate-random", vs_random.e1_win_rate, step)
            tb_writer.add_scalar(f"game-len-random", vs_random.avg_moves, step)
            tb_writer.flush()

            out_path = args.out_dir / f"checkpoint_{step}.pt"
            logger.info("saved checkpoint", out_path=out_path)
            torch.save(cpu_online_model.state_dict(), out_path)
            start_time = time.perf_counter()
            play_time = 0.0
            opt_time = 0.0
