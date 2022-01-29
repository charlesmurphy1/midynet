#!/bin/bash

echo "Loading git-submodules"
git submodule init
git submodule update

echo "Installing basegraph==1.0.0"
cd _midynet/base_graph
[ ! -d "./build" ] && mkdir build
cd build
cmake ..
make -j4
cd ..
pip install .

echo "Installing SamplableSet==2.2.0"
cd ../SamplableSet/src
[ ! -d "./build" ] && mkdir build
cd build
cmake ..
make -j4
cd ../..
pip install .

echo "Installing midynet==0.0.1"
cd ../../
[ -d "./build" ] && rm -r build
pip install -e .
