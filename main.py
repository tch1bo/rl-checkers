from pydantic import BaseModel
from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from lib.benchmark import BenchmarkArgs
from lib.dqn import TrainArgs as DQNTrainArgs
from lib.play import PlayArgs


class DebugArgs(BaseModel):
    def cli_cmd(self) -> None:
        print("debug")


class CliArgs(BaseSettings):
    train_dqn: CliSubCommand[DQNTrainArgs]
    benchmark: CliSubCommand[BenchmarkArgs]
    play: CliSubCommand[PlayArgs]
    debug: CliSubCommand[DebugArgs]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CliArgs)


if __name__ == "__main__":
    main()
