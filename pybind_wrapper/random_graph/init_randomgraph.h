#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_randomgraph_baseclass.h"

void initRandomGraph(pybind11::module& m){
    initRandomGraphBaseClass(m);
}

#endif
