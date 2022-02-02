import argparse

import midynet
from midynet.util.loggers import LoggerDict, MemoryLogger, TimeLogger


def main():
    parser = argparse.ArgumentParser(
        description="Run a meta-experiment for midyet."
    )
    parser.add_argument(
        "--path_to_config",
        "-c",
        type=str,
        metavar="PATH_TO_CONFIG",
        help="Path to the config file to define the experiment.",
        required=True,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        metavar="VERBOSE",
        required=False,
        default=0,
    )

    args = parser.parse_args()
    config = midynet.config.Config.load(args.path_to_config)
    exp = midynet.experiments.Experiment(
        config,
        args.verbose,
        loggers=LoggerDict(time=TimeLogger(), memory=MemoryLogger()),
    )
    exp.run()


if __name__ == "__main__":
    main()
