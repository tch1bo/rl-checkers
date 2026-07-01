from dataclasses import dataclass
from typing import override

import torch
from draughts import BaseAgent, BaseBoard, Move

from lib.models import MLPVNet


@dataclass
class AlphaBetaNet(BaseAgent):
    net: MLPVNet
    depth: int

    def best_move_and_value(self, root_board: BaseBoard) -> tuple[Move, float]:
        # TODO(chibo): set time limits
        # TODO(chibo): prioritize the moves based on values. Batch the inference for this
        self.net.eval()

        # NOTE: all the values are computed and stored from the root player's perspective
        @dataclass(slots=True)
        class StackItem:
            board: BaseBoard
            depth_left: int
            alpha: float
            beta: float
            moves: list[Move]
            child_values: list[float]

        stack: list[StackItem] = [
            StackItem(
                root_board,
                self.depth,
                alpha=float("-inf"),
                beta=float("+inf"),
                moves=root_board.legal_moves,
                child_values=[],
            )
        ]

        def state_value(board: BaseBoard) -> float:
            """Returns the value from the root player's perspective"""

            input = torch.from_numpy(board.to_tensor()).unsqueeze(0)
            value = float(self.net.forward(input).reshape(-1).to("cpu").item())
            return value if board.turn == root_board.turn else -value

        while stack:
            cur = stack[-1]

            if cur.board.game_over:
                # Handle terminal node
                value = 0.0
                if not cur.board.is_draw:
                    # The previous move was the victorious one, hence the `cur.board.turn`
                    # side lost.
                    value = -1.0 if cur.board.turn == root_board.turn else 1.0

                stack[-2].child_values.append(value)
                stack.pop()
                continue

            if cur.depth_left == 0:
                # Handle terminal node
                stack[-2].child_values.append(state_value(cur.board))
                stack.pop()
                continue

            # Recompute alpha, beta based on the last child update
            if cur.child_values:
                if cur.board.turn == root_board.turn:
                    # Maximizing node
                    new_alpha = max(cur.child_values)
                    if new_alpha >= cur.beta:
                        stack[-2].child_values.append(new_alpha)
                        stack.pop()
                        continue
                    cur.alpha = max(new_alpha, cur.alpha)

                else:
                    # Minimizing node
                    new_beta = min(cur.child_values)
                    if cur.alpha >= new_beta:
                        stack[-2].child_values.append(new_beta)
                        stack.pop()
                        continue
                    cur.beta = min(new_beta, cur.beta)

            # Expand the next child (if any)
            if len(cur.child_values) < len(cur.moves):
                next_move = cur.moves[len(cur.child_values)]
                cur.board.push(next_move)
                next_board = cur.board.copy()
                cur.board.pop()
                stack.append(
                    StackItem(
                        next_board,
                        cur.depth_left - 1,
                        cur.alpha,
                        cur.beta,
                        moves=next_board.legal_moves,
                        child_values=[],
                    )
                )
            else:
                node_value = (
                    max(cur.child_values)
                    if cur.board.turn == root_board.turn
                    else min(cur.child_values)
                )
                stack.pop()
                if not stack:
                    # `cur` was the root node and we processed all of its children, hence we're done
                    moves_and_values = zip(cur.moves, cur.child_values, strict=True)
                    return max(moves_and_values, key=lambda mv: mv[1])
                stack[-1].child_values.append(node_value)

        raise Exception("unreachable")

    @override
    def select_move(self, board: BaseBoard) -> Move:
        return self.best_move_and_value(board)[0]
