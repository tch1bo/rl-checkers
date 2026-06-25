from draughts import Benchmark
from pydantic import BaseModel
from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from lib.benchmark import BenchmarkArgs
from lib.dqn import TrainArgs as DQNTrainArgs
from lib.models import MLPVNet
from lib.offline_lambda import TrainArgs as OfflineLambdaTrainArgs
from lib.play import PlayArgs
from lib.utils import DEFAULT_BOARD, RandomEngine, choose_board_class


class DebugArgs(BaseModel):
    def cli_cmd(self) -> None:
        vnet = MLPVNet.init_with_random_weights(choose_board_class(DEFAULT_BOARD))
        stats = Benchmark(
            RandomEngine(),
            vnet.as_engine(),
            board_class=choose_board_class(DEFAULT_BOARD),
            games=1,
            workers=1,
        ).run()
        print(stats)


class CliArgs(BaseSettings):
    train_dqn: CliSubCommand[DQNTrainArgs]
    train_offline_lambda: CliSubCommand[OfflineLambdaTrainArgs]

    benchmark: CliSubCommand[BenchmarkArgs]
    play: CliSubCommand[PlayArgs]
    debug: CliSubCommand[DebugArgs]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CliArgs)


if __name__ == "__main__":
    main()
