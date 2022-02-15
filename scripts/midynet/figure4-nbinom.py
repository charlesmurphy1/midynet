from .script_util import (
    get_config_figure4Nbinom,
    PATH_TO_RUN_EXEC,
    EXECUTION_COMMAND,
    SPECS,
)

from midynet.scripts import ScriptManager


def main():
    for dynamics in ["sis", "cowan", "ising"]:
        config = get_config_figure4Nbinom(dynamics, num_procs=32, mem=12)
        script = ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
        )
        config10, config100 = script.split_param(config, "dynamics.num_steps")

        config10.resources["time"] = "2:00:00"
        script.run(
            config10,
            resources=config10.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )

        config100.resources["time"] = "6:00:00"
        script.run(
            config100,
            resources=config100.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
            teardown=False,
        )


if __name__ == "__main__":
    main()
