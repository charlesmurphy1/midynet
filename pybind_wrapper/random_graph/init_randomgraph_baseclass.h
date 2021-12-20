#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RANDOM_GRAPH_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/random_graph.h"

void initRandomGraphBaseClass(pybind11::module& m){
//     pybind11::class_<FastMIDyNet::RandomGraph>(m, "RandomGraph")
//         .def(pybind11::init<size_t>(), pybind11::arg("size"))
//         .def("getState", &FastMIDyNet::RandomGraph::getState)
//         .def("setState", &FastMIDyNet::RandomGraph::setState, pybind11::arg("state"))
//         .def("getSize", &FastMIDyNet::RandomGraph::getSize)
//         .def("sample", &FastMIDyNet::RandomGraph::sample)
//         .def("applyMove", &FastMIDyNet::RandomGraph::applyMove, pybind11::arg("move"));
}

#endif
