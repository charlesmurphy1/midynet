import argparse
import midynet


def main():
    parser = argparse.ArgumentParser(description="Run a meta-experiment for midyet.")
    parser.add_argument(
        "--path_to_config",
        "-c",
        type=str,
        metavar="PATH_TO_CONFIG",
        help="Path to the config file to define the experiment.",
        required=True,
    )

    args = parser.parse_args()
    config = midynet.config.Config.load(args.path_to_config)
    exp = midynet.Experiment(config)
    exp.clean()
    exp.run()


if __name__ == "__main__":
    main()
