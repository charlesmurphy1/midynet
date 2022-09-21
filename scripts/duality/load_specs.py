import json
import os
import pathlib

SPECS = json.load(open("specs.json", "r"))
if os.getenv("SERVER_NAME") in SPECS:
    SPECS = SPECS[os.getenv("SERVER_NAME")]
else:
    SPECS = SPECS["default"]

PATH_TO_DATA = pathlib.Path(SPECS["path_to_data"])
PATH_TO_RUN_EXEC = dict(run="python ../../midynet/scripts/run.py")
PATH_TO_LOG = pathlib.Path("./log")
if not PATH_TO_LOG.exists():
    PATH_TO_LOG.mkdir()
EXECUTION_COMMAND = SPECS["command"]

if __name__ == "__main__":
    pass
