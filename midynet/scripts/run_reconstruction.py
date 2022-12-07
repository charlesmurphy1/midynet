import os
import argparse
import pyhectiqlab
import logging

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
        required=True,
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
        os.mkdir(config.path)
    if "recon_information" not in config.metrics.metrics_names:
        msg = f"For this script, 'recon_information' must be in 'config.metrics'."
        raise ValueError(msg)

    run = pyhectiqlab.Run(args.name, project="dynamica/midynet")
    run.add_config(config)
    logger = run.add_log_stream(level=20)
    metrics = midynet.config.MetricsFactory.build(config)

    run.running()
    for k in config.metrics.metrics_names:
        logger.info(f"---Computing  {k}---")
        metrics[k].compute(config, logger=logger)
        path_to_metrics = metrics[k].to_pickle(config.path)
        run.add_artifact(path_to_metrics)
    run.completed()
    run.logs_buffer.flush_cache()
