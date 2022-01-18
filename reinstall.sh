#!/bin/bash

rm -rf build/* tmp/

verbose=${1:-"n"}
if [[ "$verbose" == "v" || "$verbose" == "verbose" ]];
    then  python setup.py develop
    else  pip install -e .
fi
