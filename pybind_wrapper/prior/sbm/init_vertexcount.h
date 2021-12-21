#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_VERTEXCOUNT_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_VERTEXCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/prior/sbm/python/vertexcount.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initVertexCountPrior(py::module& m){
    py::class_<VertexCountPrior, Prior<std::vector<size_t>>, PyVertexCountPrior<>>(m, "VertexCountPrior")
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"))
        .def("get_log_likelihood_ratio_from_graphmove", &VertexCountPrior::getLogLikelihoodRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_likelihood_ratio_from_blockmove", &VertexCountPrior::getLogLikelihoodRatioFromBlockMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_graphmove", &VertexCountPrior::getLogJointRatioFromGraphMove,
            py::arg("move"))
        .def("get_log_joint_ratio_from_blockmove", &VertexCountPrior::getLogJointRatioFromBlockMove,
            py::arg("move"))
        .def("apply_graphmove", &VertexCountPrior::applyGraphMove,
            py::arg("move"))
        .def("apply_blockmove", &VertexCountPrior::applyBlockMove,
            py::arg("move"))
        .def("get_block_count", &VertexCountPrior::getBlockCount)
        .def("get_block_count_prior", &VertexCountPrior::getBlockCountPrior);

    py::class_<VertexCountUniformPrior, VertexCountPrior>(m, "VertexCountUniformPrior")
        .def(py::init<size_t, BlockCountPrior&>(), py::arg("size"), py::arg("block_count_prior"));
}

}

#endif
