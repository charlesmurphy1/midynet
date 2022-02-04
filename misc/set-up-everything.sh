#!/bin/bash

cd ../
echo Loading git-submodules
git submodule init
git submodule update

echo Install C++ dependencies
sudo apt install gcc-9 g++ cmake

echo Installing basegraph==1.0.0
cd _midynet/base_graph
[ ! -d "./build" ] && mkdir build
cd build
cmake ..
make -j4
cd ..
pip install .

echo Installing SamplableSet==2.2.0
cd ../SamplableSet/src
[ ! -d "./build" ] && mkdir build
cd build
cmake ..
make -j4
cd ../..
pip install .

echo Build backend and tests
cd ../
[ ! -d "./build" ] && mkdir build
cd build
cmake ..
make -j4
cd ../..

echo Install dev-dependencies
pip install -r requirements_dev.txt
pip install -r requirements.txt

echo Installing midynet==0.0.1
[ -d "./build" ] && rm -r build
pip install .
