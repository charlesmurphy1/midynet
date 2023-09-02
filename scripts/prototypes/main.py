import argparse
import os
import shutil
import dotenv

from midynet.scripts import ScriptManager

dotenv.load_dotenv()
PATH_TO_VENV = os.getenv("PATH_TO_VENV", "../../venv")
EXECUTION_COMMAND = os.getenv("EXECUTION_COMMAND", "bash")
MAXNUMPROCS = int(os.getenv("MAXNUMPROCS", 4))
ACCOUNT = os.getenv("ACCOUNT", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-name",
        type=str,
    )
    parser.add_argument(
        "--time",
        default="24:00:00",
        type=str,
    )
    parser.add_argument(
        "--n-workers",
        default=MAXNUMPROCS,
        type=int,
    )
    parser.add_argument(
        "--mem",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
    )
    parser.add_argument(
        "--script-name",
        default="recon",
        type=str,
    )

    args = parser.parse_args()
    config_module = __import__(args.configs_name)
    configs = getattr(
        config_module,
        [x for x in dir(config_module) if "ScriptConfig" in x][0],
    )

    for config in configs.all(
        time=args.time, mem=args.mem, n_workers=args.n_workers
    ):
        if args.overwrite and os.path.exists(config.path):
            shutil.rmtree(config.path)
            os.makedirs(config.path)
        path_to_config = f"./configs/{config.name}.pkl"
        config.save(path_to_config)
        script = ScriptManager(
            executable=f"python ../../midynet/scripts/{args.script_name}.py",
            execution_command=os.environ["EXECUTION_COMMAND"],
            path_to_scripts="./scripts",
        )
        args = {
            "name": config.name,
            "path_to_config": path_to_config,
            "resume": args.resume,
            "save_patience": 1,
        }
        script.run(
            name=config.name,
            modules_to_load=[
                "StdEnv/2020",
                "gcc/9",
                "python/3.8",
                "graph-tool",
                "scipy-stack",
                "httpproxy",
                "mpi4py",
            ],
            virtualenv=PATH_TO_VENV,
            extra_args=args,
            resources=config.resources.dict,
        )


if __name__ == "__main__":
    main()
