import midynet
from config import *


def main():
    for dynamics in ["ising"]:
        config = get_config_figure4Nbinom(dynamics, num_procs=32, mem=12)
        script = midynet.scripts.ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
        )
        config10, config100 = script.split_param(config, "dynamics.num_steps")

        config10.resources["time"] = "10:00:00"
        script.run(
            config10,
            resources=config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )

        config100.resources["time"] = "24:00:00"
        script.run(
            config100,
            resources=config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
