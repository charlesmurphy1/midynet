import midynet
from config import *


def main():
    config = config_figure2Exact("sis", num_procs=4, time="24:00:00", mem=12)
    script = midynet.scripts.ScriptManager(
        executable=PATH_TO_RUN_EXEC["run"],
        execution_command=EXECUTION_COMMAND,
        path_to_scripts="./scripts",
    )
    script.run(
        config,
        resources=config.resources,
        modules_to_load=SPECS["modules_to_load"],
        virtualenv=SPECS["virtualenv"],
        extra_args=dict(verbose=2),
        teardown=False,
    )


if __name__ == "__main__":
    main()
