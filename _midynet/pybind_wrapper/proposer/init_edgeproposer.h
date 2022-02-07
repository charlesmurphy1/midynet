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
        .def(py::init<bool, bool>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true)
        .def("set_up", &EdgeProposer::setUp, py::arg("random_graph"))
        .def("allow_self_loops", &EdgeProposer::allowSelfLoops)
        .def("allow_multiedges", &EdgeProposer::allowMultiEdges)
        .def("get_log_proposal_ratio", &EdgeProposer::getLogProposalProbRatio, py::arg("move"))
        .def("update", py::overload_cast<const GraphMove&>(&EdgeProposer::updateProbabilities), py::arg("move"))
        .def("update", py::overload_cast<const BlockMove&>(&EdgeProposer::updateProbabilities), py::arg("move"));

    /* Double edge swap proposers */
    py::class_<DoubleEdgeSwapProposer, EdgeProposer>(m, "DoubleEdgeSwapProposer")
        .def(py::init<bool, bool>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true);

    /* Hinge flip proposers */
    py::class_<HingeFlipProposer, EdgeProposer, PyHingeFlipProposer<>>(m, "HingeFlipProposer")
        .def(py::init<bool, bool>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true)
        .def("set_vertex_sampler", &HingeFlipProposer::setVertexSampler,
            py::arg("vertex_sampler")) ;

    py::class_<HingeFlipUniformProposer, HingeFlipProposer>(m, "HingeFlipUniformProposer")
        .def(py::init<bool, bool>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true);

    py::class_<HingeFlipDegreeProposer, HingeFlipProposer>(m, "HingeFlipDegreeProposer")
        .def(py::init<bool, bool, size_t>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true, py::arg("shift")=1) ;

    /* Single edge proposers */
    py::class_<SingleEdgeProposer, EdgeProposer, PySingleEdgeProposer<>>(m, "SingleEdgeProposer")
        .def(py::init<bool, bool>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true)
        .def("set_vertex_sampler", &SingleEdgeProposer::setVertexSampler, py::arg("vertex_sampler")) ;

    py::class_<SingleEdgeUniformProposer, SingleEdgeProposer>(m, "SingleEdgeUniformProposer")
        .def(py::init<bool, bool>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true);

    py::class_<SingleEdgeDegreeProposer, SingleEdgeProposer>(m, "SingleEdgeDegreeProposer")
        .def(py::init<bool, bool, size_t>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true, py::arg("shift")=1) ;

}

}

#endif
