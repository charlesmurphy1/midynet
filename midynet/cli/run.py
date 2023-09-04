import os
import sys
import argparse
import datetime
import pyhectiqlab
import logging
import multiprocessing
import click

import midynet

from midynet.metrics import Progress, MemoryCheck, Checkpoint


@click.group()
def run_group():
    pass


@run_group.command(
    name="run",
    help="Run a midynet numerical experiment.",
)
@click.option(
    "--config-path", "-c", help="Path to the config file (.pkl).", type=str
)
@click.option(
    "--resume", "-r", help="Resume script where it was.", is_flag=True
)
@click.option(
    "--save-patience", "-p", help="Patience for saving.", default=1, type=int
)
def run_script(
    config_path: str,
    resume: bool,
    save_patience: int,
):
    metaconfig = midynet.config.Config.load(config_path)
    metaconfig.metrics.not_sequence("metrics_names")

    if not os.path.exists(metaconfig.path):
        os.makedirs(metaconfig.path)

    metrics = midynet.config.MetricsFactory.build(metaconfig)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    begin = datetime.datetime.now()

    for k in metaconfig.metrics.metrics_names:
        if logger is not None:
            logger.info(f"---Computing {metrics[k].__class__.__name__}---")
        config = metaconfig.copy()
        config.metrics = metaconfig.metrics.get(k)

        callbacks = [
            Progress.to_setup(
                logger=logger,
                total=len(config) // config.get("n_async_jobs", 1),
            ),
            MemoryCheck.to_setup("gb", logger=logger),
            Checkpoint.to_setup(
                patience=save_patience,
                savepath=config.path,
                logger=logger,
                metrics=metrics[k],
            ),
        ]
        metrics[k].compute(
            config,
            resume=resume,
            n_workers=config.get("n_workers", 1),
            n_async_jobs=config.get("n_async_jobs", 1),
            callbacks=callbacks,
        )
        for c in callbacks:
            c.teardown()

    end = datetime.datetime.now()
    logger.info(f"Total computation time: {end - begin}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run a reconstruction numerical experiment."
#     )

#     parser.add_argument(
#         "--run",
#         "-r",
#         metavar="RUN",
#         help="Name of the run.",
#         nargs="*",
#         default=None,
#     )
#     parser.add_argument(
#         "--name",
#         "-n",
#         metavar="NAME",
#         help="Name of the generated data.",
#         default=None,
#     )
#     parser.add_argument(
#         "--version",
#         "-v",
#         metavar="VERSION",
#         help="Version of the generated data.",
#         default=None,
#     )
#     parser.add_argument(
#         "--path_to_config",
#         "-c",
#         type=str,
#         metavar="PATH_TO_CONFIG",
#         help="Path to the config file (.pkl).",
#         required=True,
#     )
#     parser.add_argument(
#         "--path_to_credentials",
#         "-d",
#         type=str,
#         metavar="PATH_TO_CREDENTIALS",
#         help="Path to the hectiq lab credential file (usually /HOME/.hectiqlab/credentials).",
#         default=None,
#     )
#     parser.add_argument(
#         "--resume",
#         action="store_true",
#     )
#     parser.add_argument(
#         "--push_data",
#         action="store_true",
#     )
#     parser.add_argument(
#         "--save_patience",
#         type=int,
#         default=5,
#     )
#     multiprocessing.set_start_method("spawn")

#     args = parser.parse_args()
#     metaconfig = midynet.config.Config.load(args.path_to_config)
#     metaconfig.metrics.not_sequence("metrics_names")

#     if not os.path.exists(metaconfig.path):
#         os.makedirs(metaconfig.path)

#     metrics = midynet.config.MetricsFactory.build(metaconfig)
#     if args.path_to_credentials is not None:
#         os.environ["HECTIQLAB_CREDENTIALS"] = args.path_to_credentials
#     else:
#         os.environ["HECTIQLAB_CREDENTIALS"] = os.path.join(
#             os.path.expanduser("~"), ".hectiqlab/credentials"
#         )

#     if args.run is not None:
#         try:
#             run = pyhectiqlab.Run(
#                 " ".join(args.run), project="dynamica/midynet"
#             )
#             run.clear_logs()
#             run.add_meta("command-line-args", value=" ".join(sys.argv))
#             run.add_config(metaconfig)
#             logger = run.add_log_stream(level=20)
#             run.add_meta(
#                 key="Start evaluation",
#                 value=datetime.datetime.now().strftime(
#                     "%A, %d %B %Y at %H:%M:%S"
#                 ),
#             )
#             run.running()
#         except:
#             run = None
#     else:
#         run = None
#     if run is None:
#         logger = logging.getLogger()
#         logger.setLevel(logging.DEBUG)
#         handler = logging.StreamHandler(sys.stdout)
#         handler.setLevel(logging.INFO)
#         formatter = logging.Formatter(
#             "%(asctime)s - %(levelname)s - %(message)s"
#         )
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)

#     begin = datetime.datetime.now()
#     print(metaconfig)

#     for k in metaconfig.metrics.metrics_names:
#         if logger is not None:
#             logger.info(f"---Computing {metrics[k].__class__.__name__}---")
#         config = metaconfig.copy()
#         config.metrics = metaconfig.metrics.get(k)

#         callbacks = [
#             Progress.to_setup(
#                 logger=logger,
#                 total=len(config) // config.get("n_async_jobs", 1),
#             ),
#             MemoryCheck.to_setup("gb", logger=logger),
#             Checkpoint.to_setup(
#                 patience=args.save_patience,
#                 savepath=config.path,
#                 logger=logger,
#                 metrics=metrics[k],
#             ),
#         ]
#         metrics[k].compute(
#             config,
#             resume=args.resume,
#             n_workers=config.get("n_workers", 1),
#             n_async_jobs=config.get("n_async_jobs", 1),
#             callbacks=callbacks,
#         )

#         if run is not None and args.push_data:
#             run.add_artifact(
#                 os.path.join(config.path, metrics[k].shortname + ".pkl")
#             )

#         for c in callbacks:
#             c.teardown()

#     end = datetime.datetime.now()
#     logger.info(f"Total computation time: {end - begin}")

#     if args.name is not None and run is not None and args.push_data:
#         try:
#             run.add_dataset(
#                 config.path,
#                 name=args.name,
#                 version=args.version,
#                 push_dir=True,
#             )
#         except:
#             run.add_dataset(
#                 config.path,
#                 name=args.name,
#                 version=args.version,
#                 push_dir=True,
#                 resume_upload=True,
#             )

#     if run is not None:
#         run.add_meta(
#             key="End evaluation",
#             value=datetime.datetime.now().strftime(
#                 "%A, %d %B %Y at %H:%M:%S"
#             ),
#         )
#         run.completed()
#         run.logs_buffer.flush_cache()
