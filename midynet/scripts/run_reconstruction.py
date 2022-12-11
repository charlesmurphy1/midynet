import os
import sys
import argparse
import datetime
import pyhectiqlab

import midynet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a reconstruction numerical experiment."
    )

    parser.add_argument(
        "--name",
        "-n",
        metavar="RUN_NAME",
        help="Name of the run.",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--path_to_config",
        "-c",
        type=str,
        metavar="PATH_TO_CONFIG",
        help="Path to the config file (.pkl).",
        required=True,
    )

    args = parser.parse_args()
    config = midynet.config.Config.load(args.path_to_config)

    if not os.path.exists(config.path):
        os.makedirs(config.path)
    if "recon_information" not in config.metrics.metrics_names:
        msg = f"For this script, 'recon_information' must be in 'config.metrics'."
        raise ValueError(msg)

    metrics = midynet.config.MetricsFactory.build(config)
    if args.name is not None:
        run = pyhectiqlab.Run(" ".join(args.name), project="dynamica/midynet")
        run.add_meta("command-line-args", value=" ".join(sys.argv))
        run.add_config(config)
        logger = run.add_log_stream(level=20)
        run.add_meta(
            key="Start evaluation",
            value=datetime.datetime.now().strftime(
                "%A, %d %B %Y at %H:%M:%S"
            ),
        )
        run.running()
    else:
        run = None
        logger = "stdout"

    for k in config.metrics.metrics_names:
        metrics[k].compute(config, logger=logger)
        path_to_metrics = metrics[k].to_pickle(config.path)
        if run is not None:
            run.add_artifact(path_to_metrics)

    if run is not None:
        run.add_meta(
            key="End evaluation",
            value=datetime.datetime.now().strftime(
                "%A, %d %B %Y at %H:%M:%S"
            ),
        )
        run.completed()
        run.logs_buffer.flush_cache()
