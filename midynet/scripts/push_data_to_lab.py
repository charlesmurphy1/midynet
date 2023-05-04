import os
import pyhectiqlab
import argparse

from typing import Optional


def main(*, path_to_data: str, run: str, name: str, version: Optional[str] = None):
    os.environ["HECTIQLAB_CREDENTIALS"] = os.path.join(
        os.path.expanduser("~"), ".hectiqlab/credentials"
    )
    run = pyhectiqlab.Run(run, project="dynamica/midynet")

    for root, _, files in os.walk(path_to_data):
        for f in files:
            run.add_artifact(os.path.join(root, f))
        break

    run.add_dataset(path_to_data, name, version=version, push_dir=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        "-p",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        required=True,
        nargs="*",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=None,
        required=False,
    )

    args = parser.parse_args()

    args.run = " ".join(args.run)
    main(**args.__dict__)
