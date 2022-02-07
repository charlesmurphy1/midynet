#ifndef FAST_MIDYNET_PYWRAPPER_INIT_EdgeSAMPLER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_EdgeSAMPLER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/edge_sampler.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initVertexSampler(py::module& m){
    py::class_<EdgeSampler>(m, "EdgeSampler")
        .def(py::init<>())
        .def("sample", &EdgeSampler::sample)
        .def("set_up", &EdgeSampler::setUp, py::arg("graph"))
        .def("add_edge", &EdgeSampler::addEdge, py::arg("edge"))
        .def("remove_edge", &EdgeSampler::removeEdge, py::arg("edge"))
        .def("insert_edge", &EdgeSampler::insertEdge, py::arg("edge"), py::arg("weight"))
        .def("erase_edge", &EdgeSampler::eraseEdge, py::arg("edge"))
        .def("get_vertex_weight", &EdgeSampler::getVertexWeight, py::arg("vertex"))
        .def("get_edge_weight", &EdgeSampler::getEdgeWeight, py::arg("edge"))
        .def("get_total_weight", &EdgeSampler::getTotalWeight)
        .def("get_size", &EdgeSampler::getTotalSize)
        .def("check_safety", &EdgeSampler::checkSafety)
        .def("clear", &EdgeSampler::clear)
        ;

}

}

#endif
