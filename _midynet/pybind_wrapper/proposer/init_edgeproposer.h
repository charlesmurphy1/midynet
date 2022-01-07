#ifndef FAST_MIDYNET_PYWRAPPER_INIT_EDGEPROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_EDGEPROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/proposer/python/proposer.hpp"
#include "FastMIDyNet/proposer/edge_proposer/double_edge_swap.h"
#include "FastMIDyNet/proposer/edge_proposer/hinge_flip.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initEdgeProposer(py::module& m){
    py::class_<EdgeProposer, Proposer<GraphMove>, PyEdgeProposer<>>(m, "EdgeProposer")
        .def(py::init<>())
        .def("set_up", &EdgeProposer::setUp, py::arg("random_graph"))
        .def("accept_isolated", &EdgeProposer::setAcceptIsolated, py::arg("accept"))
        .def("accept_isolated", &EdgeProposer::getAcceptIsolated);

    /* Double edge swap proposers */
    py::class_<DoubleEdgeSwapProposer, EdgeProposer>(m, "DoubleEdgeSwapProposer")
        .def(py::init<>()) ;

    /* Hinge flip proposers */
    py::class_<HingeFlipProposer, EdgeProposer>(m, "HingeFlipProposer")
        .def(py::init<>())
        .def("set_vertex_sampler", &HingeFlipProposer::setVertexSampler,
            py::arg("vertex_sampler")) ;

    py::class_<HingeFlipUniformProposer, HingeFlipProposer>(m, "HingeFlipUniformProposer")
        .def(py::init<>()) ;

    py::class_<HingeFlipDegreeProposer, HingeFlipProposer>(m, "HingeFlipDegreeProposer")
        .def(py::init<size_t>(), py::arg("shift")=1) ;

    /* Single edge proposers */
    py::class_<SingleEdgeProposer, EdgeProposer>(m, "SingleEdgeProposer")
        .def(py::init<>())
        .def("set_vertex_sampler", &SingleEdgeProposer::setVertexSampler,
            py::arg("vertex_sampler")) ;

    py::class_<SingleEdgeUniformProposer, SingleEdgeProposer>(m, "SingleEdgeUniformProposer")
        .def(py::init<>()) ;

    py::class_<SingleEdgeDegreeProposer, SingleEdgeProposer>(m, "SingleEdgeDegreeProposer")
        .def(py::init<size_t>(), py::arg("shift")=1) ;
}

}

#endif
