import os
import sys
import argparse
import datetime
import pyhectiqlab
import logging
import multiprocessing

import midynet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a reconstruction numerical experiment."
    )

    parser.add_argument(
        "--run",
        "-r",
        metavar="RUN",
        help="Name of the run.",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--name",
        "-n",
        metavar="NAME",
        help="Name of the generated data.",
        default=None,
    )
    parser.add_argument(
        "--version",
        "-v",
        metavar="VERSION",
        help="Version of the generated data.",
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
    parser.add_argument(
        "--path_to_credentials",
        "-d",
        type=str,
        metavar="PATH_TO_CREDENTIALS",
        help="Path to the hectiq lab credential file (usually /HOME/.hectiqlab/credentials).",
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    multiprocessing.set_start_method("spawn")

    args = parser.parse_args()
    metaconfig = midynet.config.Config.load(args.path_to_config)
    metaconfig.metrics.not_sequence("metrics_names")

    if not os.path.exists(metaconfig.path):
        os.makedirs(metaconfig.path)

    metrics = midynet.config.MetricsFactory.build(metaconfig)
    if args.path_to_credentials is not None:
        os.environ["HECTIQLAB_CREDENTIALS"] = args.path_to_credentials
    else:
        os.environ["HECTIQLAB_CREDENTIALS"] = os.path.join(
            os.path.expanduser("~"), ".hectiqlab/credentials"
        )

    if args.run is not None:
        try:
            run = pyhectiqlab.Run(
                " ".join(args.run), project="dynamica/midynet"
            )
            run.clear_logs()
            run.add_meta("command-line-args", value=" ".join(sys.argv))
            run.add_config(metaconfig)
            logger = run.add_log_stream(level=20)
            run.add_meta(
                key="Start evaluation",
                value=datetime.datetime.now().strftime(
                    "%A, %d %B %Y at %H:%M:%S"
                ),
            )
            run.running()
        except:
            run = None
    if run is None:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    begin = datetime.datetime.now()

    for k in metaconfig.metrics.metrics_names:
        begin_metrics = datetime.datetime.now()
        if logger is not None:
            logger.info(f"---Computing {metrics[k].__class__.__name__}---")
        config = metaconfig.copy()
        config.metrics = metaconfig.metrics.get(k)
        metrics[k].compute(
            config, logger=logger, resume=args.resume, save_path=config.path
        )
        end_metrics = datetime.datetime.now()
        if logger is not None:
            logger.info(
                f"---Finish with time: {end_metrics - begin_metrics}---"
            )

        if run is not None:
            run.add_artifact(
                os.path.join(config.path, metrics[k].shortname + ".pkl")
            )

    end = datetime.datetime.now()
    logger.info(f"Total computation time: {begin - end}")

    if args.name is not None and run is not None:
        try:
            run.add_dataset(
                config.path,
                name=args.name,
                version=args.version,
                push_dir=True,
            )
        except:
            run.add_dataset(
                config.path,
                name=args.name,
                version=args.version,
                push_dir=True,
                resume_upload=True,
            )

    if run is not None:
        run.add_meta(
            key="End evaluation",
            value=datetime.datetime.now().strftime(
                "%A, %d %B %Y at %H:%M:%S"
            ),
        )
        run.completed()
        run.logs_buffer.flush_cache()
