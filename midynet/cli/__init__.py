import sys
import click

from .configure import configure_group
from .run import run_group
from .launch import launch_group

# from .visualize import visualize_group


def main():
    cli = click.CommandCollection(
        sources=[
            configure_group,
            run_group,
            launch_group,
            # visualize_group,
        ]
    )
    # Standalone mode is False so that the errors can be caught by the runs
    cli(standalone_mode=False)
    sys.exit()


if __name__ == "__main__":
    main()
