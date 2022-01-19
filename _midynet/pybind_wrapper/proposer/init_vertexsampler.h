#ifndef FAST_MIDYNET_PYWRAPPER_INIT_VERTEXSAMPLER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_VERTEXSAMPLER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/edge_proposer/vertex_sampler.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initVertexSampler(py::module& m){
    py::class_<VertexSampler, PyVertexSampler<>>(m, "VertexSampler")
        .def(py::init<>())
        .def("sample", &VertexSampler::sample)
        .def("set_up", &VertexSampler::setUp, py::arg("graph"))
        .def("update", py::overload_cast<const GraphMove&>(&VertexSampler::update), py::arg("move"))
        .def("update", py::overload_cast<const BlockMove&>(&VertexSampler::update), py::arg("move"))
        .def("get_vertex_weight", &VertexSampler::getVertexWeight, py::arg("vertex"))
        .def("get_total_weight", &VertexSampler::getTotalWeight)
        .def("accept_isolated", &VertexSampler::setAcceptIsolated, py::arg("accept"))
        .def("accept_isolated", &VertexSampler::getAcceptIsolated)
        .def("check_safety", &VertexSampler::checkSafety)
        ;

    py::class_<VertexUniformSampler, VertexSampler>(m, "VertexUniformSampler")
        .def(py::init<>());

    py::class_<VertexDegreeSampler, VertexSampler>(m, "VertexDegreeSampler")
        .def(py::init<size_t>(), py::arg("shift")=1);
}

}

#endif
