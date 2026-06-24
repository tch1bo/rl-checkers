from pathlib import Path
from typing import override

import torch
from draughts import BaseAgent, BaseBoard, Move
from torch import Tensor, nn


class MLPQNet(nn.Module, BaseAgent):
    mlp: nn.Sequential

    def __init__(self, mlp: nn.Sequential, device: torch.device) -> None:
        super().__init__()
        self.mlp = mlp.to(device)

    @staticmethod
    def init_with_random_weights(
        board_class: type[BaseBoard], device: torch.device = torch.device("cpu")
    ) -> "MLPQNet":
        n = board_class.SQUARES_COUNT
        mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n * n),
        )

        def f(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        mlp.apply(f)
        return MLPQNet(mlp, device)

    @staticmethod
    def load_from_path(
        path: Path,
        board_class: type[BaseBoard],
        device: torch.device = torch.device("cpu"),
    ) -> "MLPQNet":
        qnet = MLPQNet.init_with_random_weights(board_class, device)
        qnet.load_state_dict(torch.load(path, map_location=device), strict=True)
        return qnet

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x.to(device=self.device))

    @override
    def select_move(self, board: BaseBoard) -> Move:
        self.eval()

        input = torch.from_numpy(board.to_tensor()).unsqueeze(0)
        predictions = self.forward(input).reshape(-1).to("cpu")
        predictions[~board.legal_moves_mask()] = float("-inf")
        move_index = predictions.argmax().item()
        assert isinstance(move_index, int)
        return board.index_to_move(move_index)
