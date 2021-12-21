#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_randomgraph.h"
#include "init_sbm.h"
#include "init_dcsbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initRandomGraph(py::module& m){
    initRandomGraphBaseClass(m);
    initStochasticBlockModelFamily(m);
    initErdosRenyiFamily(m);
    initDegreeCorrectedStochasticBlockModelFamily(m);
}

}

#endif
