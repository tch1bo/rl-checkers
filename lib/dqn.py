import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Type

import numpy as np
import torch
import tqdm
from draughts import BaseBoard
from draughts import Move as DraughtsMove
from pydantic import BaseModel, Field, PrivateAttr
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from lib.log import get_logger
from lib.models import NUM_CHANNELS, MLPQNet
from lib.utils import (
    DEFAULT_BOARD,
    BoardClassLiteral,
    benchmark_against_ab_engine,
    benchmark_against_random,
    choose_board_class,
)

logger = get_logger()


class TrainArgs(BaseModel):
    """This trainer implements a modification of the DQN paper"""

    board_type: BoardClassLiteral = DEFAULT_BOARD

    learning_rate: float = Field(ge=0.0, default=1e-4)
    gamma: float = Field(ge=0.0, le=1.0, default=0.99)

    max_moves_per_game: int = Field(ge=0, default=1000)
    min_replay_buffer: int = Field(
        ge=0,
        default=1000,
        description="only start training when the replay buffer has that many samples",
    )
    env_steps_per_gradient_step: int = Field(ge=1, default=10)
    max_replay_buffer: int = Field(ge=0, default=100_000)
    eps_min: float = Field(ge=0, le=1.0, default=0.1)
    eps_decay_ratio: float = Field(
        ge=0,
        le=1.0,
        default=0.3,
        description="eps will be decayed from 1.0 to its min over this ratio of epochs",
    )
    num_steps: int = Field(ge=0, default=100000)
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


def eps_greedy_move(
    args: TrainArgs, eps: float, qnet: MLPQNet, board: BaseBoard
) -> DraughtsMove:
    if args.rng.random() > eps:
        # Greedy move
        return qnet.select_move(board)

    # Exploration move
    moves = board.legal_moves
    return moves[args.rng.integers(len(moves))]


class ReplayBuffer:
    def __init__(self, max_capacity: int, board_class: Type[BaseBoard]) -> None:
        # TODO(chibo): research pinned memory

        # The enconding of the pre and post states (NUM_CHANNELS, SQUARES_COUNT)
        self.pre_input = torch.empty(
            (max_capacity, NUM_CHANNELS, board_class.SQUARES_COUNT), dtype=torch.float32
        )
        self.post_input = torch.empty(
            (max_capacity, NUM_CHANNELS, board_class.SQUARES_COUNT), dtype=torch.float32
        )

        # The index of the move that was performed from the pre state
        self.move_index = torch.empty(max_capacity, dtype=torch.int32)

        # The mask for moves available from the post state (SQUARES_COUNT**2)
        self.post_action_mask = torch.empty(
            (max_capacity, board_class.SQUARES_COUNT**2), dtype=torch.bool
        )

        # `reward` is:
        #   1 - if the move was the last move of the winning side
        #  -1 - if the move was the last move of the losing side
        #   0 - otherwise
        self.reward = torch.empty(max_capacity, dtype=torch.int32)

        # `is_final` is True iff there were no more moves for this player after this move
        self.is_final = torch.empty(max_capacity, dtype=torch.bool)

        self.head = -1
        self.max_capacity = max_capacity
        self.is_full = False

    def append(self) -> int:
        self.head = self.head + 1
        if self.head == self.max_capacity:
            self.head = 0
            self.is_full = True
        return self.head

    def sample(self, rng: np.random.Generator, num_samples: int) -> np.ndarray:
        if self.is_full:
            indices = np.arange(self.max_capacity)
        else:
            indices = np.arange(self.head)
        return rng.choice(indices, num_samples)

    def get_samples_for_q_val_validation(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        non_final_mask = ~self.is_final[: self.head]
        return (
            self.post_input[: self.head][non_final_mask].to(device),
            self.post_action_mask[: self.head][non_final_mask].to(device),
        )


def play_one_game(
    args: TrainArgs,
    qnet: MLPQNet,
    board_class: Type[BaseBoard],
    replay_buffer: ReplayBuffer,
    eps: float,
) -> int:
    board = board_class()
    indices: list[int] = []
    qnet.eval()

    while not board.game_over and len(indices) < args.max_moves_per_game:
        index = replay_buffer.append()
        indices.append(index)
        state = board.to_tensor()

        # Store the pre_input
        replay_buffer.pre_input[index].copy_(torch.from_numpy(state))

        # Store the post input and the post mask for the previous move from the same player
        if len(indices) > 2:
            prev_move_index = indices[-3]
            replay_buffer.post_input[prev_move_index].copy_(torch.from_numpy(state))
            board.legal_moves
            replay_buffer.post_action_mask[prev_move_index].copy_(
                torch.from_numpy(board.legal_moves_mask())
            )

        # Store the reward and the is_final.
        # These are 0/false for all moves except the last two, which we'll handle separately
        replay_buffer.reward[index] = 0
        replay_buffer.is_final[index] = False

        # Choose the move and store it
        with torch.no_grad():
            move = eps_greedy_move(args, eps, qnet, board)
        replay_buffer.move_index[index] = board.move_to_index(move)
        board.push(move)

    # Handle the last two moves. We don't bother updating the `post_input` / `post_action_mask`,
    # because they will be ignored during the training loop (because `is_final == True`).
    replay_buffer.is_final[indices[-1]] = True
    replay_buffer.is_final[indices[-2]] = True

    if board.game_over and not board.is_draw:
        # The last player to make a move won
        replay_buffer.reward[indices[-1]] = 1
        # The other player lost
        replay_buffer.reward[indices[-2]] = -1

    return len(indices)


def optimize(
    args: TrainArgs,
    online_model: MLPQNet,
    target_model: MLPQNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    num_batches: int,
) -> float:
    all_indices = replay_buffer.sample(args.rng, num_batches * args.train_batch_size)

    # Compute the bootstrap term `max Q(S', a)`
    target_model.eval()
    with torch.no_grad():
        post_input = replay_buffer.post_input[all_indices].to(
            device=target_model.device
        )
        mask = replay_buffer.post_action_mask[all_indices].to(
            device=target_model.device
        )
        is_final = replay_buffer.is_final[all_indices].to(device=target_model.device)
        reward = replay_buffer.reward[all_indices].to(device=target_model.device)

        # This computes `max Q(S', a)` for all non-final moves. For the final move, the result is 0
        max_values = (
            target_model.forward(post_input)
            .masked_fill(~mask, float("-inf"))
            .amax(dim=1)
            .masked_fill(is_final, 0.0)
        )

        targets = args.gamma * max_values + reward

    # Do batched SGD
    online_model.train()
    losses: list[float] = []
    for batch_start in range(0, len(all_indices), args.train_batch_size):
        batch_indices = all_indices[batch_start : batch_start + args.train_batch_size]
        batch_targets = targets[batch_start : batch_start + args.train_batch_size]

        pre_input = replay_buffer.pre_input[batch_indices].to(online_model.device)
        move_index = replay_buffer.move_index[batch_indices].to(online_model.device)

        y = online_model.forward(pre_input)
        y = y[torch.arange(y.size(0), device=online_model.device), move_index]
        loss = mse_loss(y, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


def get_mean_max_q_values(
    qnet: MLPQNet, inputs: torch.Tensor, mask: torch.Tensor
) -> float:
    return (
        qnet.forward(inputs).masked_fill(~mask, float("-inf")).amax(dim=1).mean().item()
    )


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
    replay_buffer = ReplayBuffer(args.max_replay_buffer, board_class)
    while replay_buffer.head < args.min_replay_buffer:
        play_one_game(args, online_model, board_class, replay_buffer, 1.0)
    qval_validation_input, qval_validation_mask = (
        replay_buffer.get_samples_for_q_val_validation(online_model.device)
    )

    logger.info(f"using {len(qval_validation_input)} states for Q-value validation")

    # Do the "play -> gradient update" steps
    eps_decay_steps = max(int(args.eps_decay_ratio * args.num_steps), 1)
    optimizer = torch.optim.RMSprop(online_model.parameters(), lr=args.learning_rate)
    loss_sum: float = 0
    start_time = time.perf_counter()
    num_gradient_updates = 0
    for step in tqdm.trange(args.num_steps):
        eps = max(args.eps_min, 1.0 - step / eps_decay_steps)

        # Play one game
        num_new_moves = play_one_game(
            args, online_model, board_class, replay_buffer, eps
        )

        # Do a gradient step
        if num_new_moves > 0:
            num_batches = num_new_moves // args.env_steps_per_gradient_step
            loss = optimize(
                args, online_model, target_model, optimizer, replay_buffer, num_batches
            )
            loss_sum += loss
            tb_writer.add_scalar("loss", loss, step)

            num_gradient_updates += num_batches
            tb_writer.add_scalar("num_gradient_updates", num_gradient_updates, step)

            mean_max_q = get_mean_max_q_values(
                online_model, qval_validation_input, qval_validation_mask
            )
            tb_writer.add_scalar("mean_max_q", mean_max_q, step)

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
                num_gradient_updates=num_gradient_updates,
            )
            loss_sum = 0.0

            # Benchmark against an alpha-beta engine
            logger.info("running benchmarks against AB and random engines")
            cpu_online_model = deepcopy(online_model).to("cpu").eval()
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
