#ifndef FAST_MIDYNET_PYWRAPPER_INIT_VERTEXSAMPLER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_VERTEXSAMPLER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/python/sampler.hpp"

#include "BaseGraph/types.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "FastMIDyNet/proposer/sampler/label_sampler.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initSampler(py::module& m){
    py::class_<VertexSampler, PyVertexSampler<>>(m, "VertexSampler")
        .def(py::init<>())
        .def("sample", &VertexSampler::sample)
        .def("contains", &VertexSampler::contains, py::arg("vertex"))
        .def("on_vertex_insertion", &VertexSampler::onVertexInsertion)
        .def("on_vertex_erasure", &VertexSampler::onVertexErasure)
        .def("on_edge_insertion", &VertexSampler::onEdgeInsertion, py::arg("edge"), py::arg("weight"))
        .def("on_edge_erasure", &VertexSampler::onEdgeErasure, py::arg("edge"))
        .def("on_edge_addition", &VertexSampler::onEdgeAddition, py::arg("edge"))
        .def("on_edge_removal", &VertexSampler::onEdgeRemoval, py::arg("edge"))
        .def("get_vertex_weight", &VertexSampler::getVertexWeight, py::arg("vertex"))
        .def("get_total_weight", &VertexSampler::getTotalWeight)
        .def("get_size", &VertexSampler::getSize)
        .def("check_safety", &VertexSampler::checkSafety)
        .def("clear", &VertexSampler::clear)
        ;

    py::class_<VertexUniformSampler, VertexSampler>(m, "VertexUniformSampler")
        .def(py::init<>());

    py::class_<VertexDegreeSampler, VertexSampler>(m, "VertexDegreeSampler")
        .def(py::init<size_t>(), py::arg("shift")=1);

    py::class_<EdgeSampler>(m, "EdgeSampler")
        .def(py::init<>())
        .def("sample", &EdgeSampler::sample)
        .def("contains", &EdgeSampler::contains, py::arg("edge"))
        .def("on_edge_insertion", &EdgeSampler::onEdgeInsertion, py::arg("edge"), py::arg("weight"))
        .def("on_edge_erasure", &EdgeSampler::onEdgeErasure, py::arg("edge"))
        .def("on_edge_addition", &EdgeSampler::onEdgeAddition, py::arg("edge"))
        .def("on_edge_removal", &EdgeSampler::onEdgeRemoval, py::arg("edge"))
        .def("get_total_weight", &EdgeSampler::getTotalWeight)
        .def("get_size", &EdgeSampler::getSize)
        .def("check_safety", &EdgeSampler::checkSafety)
        .def("clear", &EdgeSampler::clear)
        ;

    // py::class_<LabelPairSampler>(m, "LabelPairSampler")
    //     .def(py::init<>())
    //     .def("sample", &LabelPairSampler::sample)
    //     .def("set_up", &LabelPairSampler::setUp, py::arg("random_graph"))
    //     .def("on_edge_insertion", &LabelPairSampler::onEdgeInsertion, py::arg("edge"), py::arg("weight"))
    //     .def("on_edge_erasure", &LabelPairSampler::onEdgeErasure, py::arg("edge"))
    //     .def("on_edge_addition", &LabelPairSampler::onEdgeAddition, py::arg("edge"))
    //     .def("on_edge_removal", &LabelPairSampler::onEdgeRemoval, py::arg("edge"))
    //     // .def("get_label_of_index", py::overload_cast<const BaseGraph::VertexIndex&>(&LabelPairSampler::getLabelOfIdx), py::arg("vertex"))
    //     // .def("get_label_of_index", py::overload_cast<const BaseGraph::Edge&>(&LabelPairSampler::getLabelOfIdx), py::arg("edge"))
    //     .def("get_label_pair_weight", &LabelPairSampler::getLabelPairWeight, py::arg("label_pair"))
    //     .def("get_vertex_total_weight", &LabelPairSampler::getVertexTotalWeight)
    //     .def("get_edge_total_weight", &LabelPairSampler::getEdgeTotalWeight)
    //     .def("get_total_weight", &LabelPairSampler::getTotalWeight)
    //     .def("check_safety", &LabelPairSampler::checkSafety)
    //     .def("clear", &LabelPairSampler::clear)
    //     ;
}

}

#endif
