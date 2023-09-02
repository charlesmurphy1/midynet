import os
import click


@click.group()
def configure_group():
    pass


@configure_group.command(
    name="configure",
    help="Configure the environment for midynet numerical experiments.",
)
@click.option(
    "--data-path",
    "-d",
    help="Path to the data folder.",
    default=None,
    type=str,
)
@click.option(
    "--n-workers",
    "-w",
    help="Number of workers.",
    default=None,
    type=int,
)
@click.option(
    "--memory",
    "-m",
    help="Available RAM memory.",
    default=0,
    type=int,
)
@click.option(
    "--execution-command",
    "-e",
    help="Execution command.",
    default=None,
    type=str,
)
def configure_environment(
    data_path: str,
    n_workers: int,
    memory: int,
    execution_command: str,
    **kwargs,
):
    HOME = os.path.expanduser("~")
    print("Configuring MiDyNet environment...")
    print(f"Found home directory at `{HOME}`.")
    if data_path is None:
        data_path = input(
            "Enter the path to the data folder [Default: '~/midynet-data']: "
        )
        if data_path == "":
            data_path = os.path.join(HOME, "midynet-data")
        os.makedirs(data_path, exist_ok=True)
    if n_workers is None:
        n_workers = input("Enter the number of workers [Default: 1]: ")
        if n_workers == "":
            n_workers = 1
        else:
            n_workers = int(n_workers)
    if memory is None:
        memory = input("Enter the available RAM memory in GB [Default: 0]: ")
        if memory == "":
            memory = 0
        else:
            memory = int(memory)
    if execution_command is None:
        execution_command = input(
            "Enter the execution command [Default: 'bash']: "
        )
        if execution_command == "":
            execution_command = "bash"
    extra = dict()
    while True:
        key = input("Enter an extra environment variable (or 'done'): ")
        if key == "":
            break
        value = input(f"Enter the value for {key}: ")
        extra[key] = value
    env_path = os.path.join(HOME, ".md-env")
    print(f"Writing configuration file at `{env_path}`...")
    with open(env_path, "w") as f:
        f.write(f"MD-DATA_PATH={data_path}\n")
        f.write(f"MD-N_WORKERS={n_workers}\n")
        f.write(f"MD-MEMORY={memory}\n")
        f.write(f"MD-EXECUTION_COMMAND={execution_command}\n")
        for key, value in extra.items():
            f.write(f"MD-{key.upper()}={value}\n")
    print("Done.")


@configure_group.command(
    name="delete-config",
    help="Deletes the configuration of the environment for midynet numerical experiments.",
)
def delete_config_script():
    os.remove("~/.md-env")
    print(".env file deleted.")


@configure_group.command(
    name="show-config",
    help="Shows the configuration of the environment for midynet numerical experiments.",
)
def show_configure_script():
    import dotenv

    HOME = os.path.expanduser("~")
    env_path = os.path.join(HOME, ".md-env")
    if not os.path.exists(env_path):
        print("No configuration file found.")
        return
    dotenv.load_dotenv(env_path)
    print(f"Current MiDyNet environment at `{env_path}`:")
    for k, v in os.environ.items():
        if not k.startswith("MD-"):
            continue
        print(f"-- {k} = {v}")
