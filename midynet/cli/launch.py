import os
import click
import dotenv
import shutil

from midynet.config import experiments

dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".md-env"))


class ScriptManager:
    def __init__(
        self,
        executable,
        execution_command: str = "bash",
        script_path=None,
    ):
        self.executable = executable
        self.execution_command = execution_command
        self.script_path = "./scripts" if script_path is None else script_path
        assert os.path.exists(self.script_path)

    def write_script(
        self,
        resource_prefix: str = "#SBATCH",
        modules_to_load: list[str] = None,
        virtualenv: str = None,
        resources: dict[str, str] = None,
        extra_args: dict[str, str] = None,
    ):
        script = "#!/bin/bash\n"
        resources = {} if resources is None else resources
        resources.pop("name", None)
        for k, r in resources.items():
            script += f"{resource_prefix} --{k.replace('_', '-')}={r}\n"

        script += "\n"
        if modules_to_load:
            script += f"module load {' '.join(modules_to_load)}\n"

        if virtualenv:
            script += f"source {virtualenv}\n  \n"
        script += f"{self.executable}"

        extra_args = {} if extra_args is None else extra_args
        for k, v in extra_args.items():
            if isinstance(v, bool):
                script += f" --{k}" if v else ""
            else:
                script += f" --{k} {v}"

        script += "\n \n"
        if virtualenv:
            script += "deactivate\n"
        return script

    def clear(self) -> None:
        shutil.rmtree(self.script_path)

    def run(self, name: str, **kwargs):
        path = os.path.join(self.script_path, f"{name}.sh")
        script = self.write_script(**kwargs)
        print(path)
        with open(path, "w") as f:
            f.write(script)
        path_to_script = os.path.join(self.script_path, f"{name}.sh")
        os.system(f"{self.execution_command} {path_to_script}")


@click.group()
def launch_group():
    pass


@launch_group.command(
    name="launch",
    help="Launch a midynet numerical experiment.",
)
@click.option(
    "--exp-name",
    "-e",
    help="Name of the experiment.",
    type=click.Choice(
        list(experiments.__all_configs__.keys()), case_sensitive=False
    ),
)
@click.option(
    "--time",
    "-t",
    help="Approximate time to run the experiment.",
    default="24:00:00",
    type=str,
)
@click.option(
    "--n-workers",
    "-w",
    help="Number of workers.",
    default=os.getenv("MD-N_WORKERS", 1),
    type=int,
)
@click.option(
    "--memory",
    "-m",
    help="Approximate memory to run the experiment (in GB).",
    default=os.getenv("MD-MEMORY", 0),
    type=int,
)
@click.option(
    "--n-async-jobs",
    "-a",
    help="Number of asynchronous jobs.",
    default=1,
)
@click.option(
    "--overwrite",
    "-o",
    help="Overwrite the experiment if it already exists.",
    is_flag=True,
)
@click.option(
    "--resume",
    "-r",
    help="Resume script where it was.",
    is_flag=True,
)
@click.option(
    "--script-path",
    "-s",
    help="Path to the script to launch.",
    default=os.getenv("MD-SCRIPT_PATH", "./scripts"),
    type=str,
)
@click.option(
    "--test-mode",
    help="Execute in test mode.",
    is_flag=True,
)
@click.option(
    "--no-confirm",
    help="Do not confirm before launching.",
    is_flag=True,
)
def launch_script(
    exp_name: str,
    time: str,
    n_workers: int,
    memory: int,
    n_async_jobs: int,
    overwrite: bool,
    resume: bool,
    script_path: str,
    test_mode: bool,
    no_confirm: bool,
):
    configs = experiments.__all_configs__[exp_name]
    configs = (
        configs.test
        if test_mode and "test" in configs.__dict__
        else configs.all
    )
    configs = configs(n_workers=n_workers, n_async_jobs=n_async_jobs)
    os.makedirs(script_path, exist_ok=True)
    for c in configs:
        c.resources.update(
            time=time,
            cpus_per_task=n_workers,
            job_name=c.name,
            output=f"log/{c.name}.out",
            mem_per_cpu=memory,
        )

        if overwrite and os.path.exists(c.path):
            shutil.rmtree(c.path)
            os.makedirs(c.path)
        config_path = os.path.join(script_path, f"{c.name}.pkl")
        c.save(config_path)
        script = ScriptManager(
            executable=f"midynet-cmd run",
            execution_command=os.getenv("EXECUTION_COMMAND", "bash"),
            script_path=script_path,
        )
        extra_args = {
            "config-path": config_path,
            "resume": resume,
            "save-patience": 1,
        }
        print(f"Launching script with config: {c.name}")
        print(c)

        if (
            not no_confirm
            and input("Press `Enter` to continue or `S` to skip...").upper()
            == "S"
        ):
            print("Skipping...")
            continue
        script.run(
            name=c.name,
            modules_to_load=os.getenv("MD-MODULES", "")
            .replace(" ", "")
            .split(","),
            virtualenv=os.getenv("MD-VIRTUALENV", None),
            extra_args=extra_args,
            resources=c.resources.dict,
        )
    print("Done")
