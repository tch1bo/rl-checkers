from pydantic import BaseModel
from pydantic_settings import BaseSettings, CliApp, CliSubCommand

from lib.benchmark import BenchmarkArgs
from lib.play import PlayArgs
from lib.trainer import TrainArgs


class DebugArgs(BaseModel):
    def cli_cmd(self) -> None:
        print("debug")


class CliArgs(BaseSettings):
    train: CliSubCommand[TrainArgs]
    benchmark: CliSubCommand[BenchmarkArgs]
    play: CliSubCommand[PlayArgs]
    debug: CliSubCommand[DebugArgs]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(CliArgs)


if __name__ == "__main__":
    main()
