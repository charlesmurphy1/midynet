import midynet
from config import *


def main():
    for dynamics in ["sis", "cowan", "ising"]:
        config = get_config_figure3Large(dynamics, num_procs=32, mem=12)
        script = midynet.scripts.ScriptManager(
            executable=PATH_TO_RUN_EXEC["run"],
            execution_command=EXECUTION_COMMAND,
            path_to_scripts="./scripts",
        )
        ais_config, mf_config = script.split_param(config, "metrics.mutualinfo.method")

        mf_config.resources["time"] = "24:00:00"
        script.run(
            mf_config,
            resources=mf_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
        )

        ais_config.resources["time"] = "48:00:00"
        script.run(
            ais_config,
            resources=ais_config.resources,
            modules_to_load=SPECS["modules_to_load"],
            virtualenv=SPECS["virtualenv"],
            extra_args=dict(verbose=2),
        )


if __name__ == "__main__":
    main()
