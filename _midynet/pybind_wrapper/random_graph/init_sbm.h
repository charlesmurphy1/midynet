#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/sbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initStochasticBlockModel(py::module& m){
    py::class_<StochasticBlockModelBase, BlockLabeledRandomGraph>(m, "StochasticBlockModelBase")
        // .def(
        //     py::init<size_t, LabelGraphPrior&, bool, bool, bool>(),
        //     py::arg("size"),
        //     py::arg("label_graph_prior"),
        //     py::arg("stub_labeled")=true,
        //     py::arg("with_self_loops")=true,
        //     py::arg("with_parallel_edges")=true
        // )
        .def("get_label_graph_prior", &StochasticBlockModelBase::getLabelGraphPrior)
        .def("set_label_graph_prior", &StochasticBlockModelBase::setLabelGraphPrior, py::arg("prior"))
        ;

    py::class_<StochasticBlockModel, StochasticBlockModelBase>(m, "StochasticBlockModel")
        .def(
            py::init<const std::vector<BlockIndex>, const LabelGraph&, bool, bool, bool, std::string>(),
            py::arg("blocks"),
            py::arg("label_graph"),
            py::arg("stub_labeled")=true,
            py::arg("with_self_loops")=true,
            py::arg("with_parallel_edges")=true,
            py::arg("edge_proposer_type")="uniform"
        )
        ;

    py::class_<StochasticBlockModelFamily, StochasticBlockModelBase>(m, "StochasticBlockModelFamily")
        .def(
            py::init<size_t, double, size_t, bool, bool, bool, bool, bool, bool, std::string, std::string, double, double, double>(),
            py::arg("size"),
            py::arg("edge_count"),
            py::arg("block_count")=0,
            py::arg("block_hyperprior")=true,
            py::arg("planted")=false,
            py::arg("canonical")=false,
            py::arg("stub_labeled")=false,
            py::arg("with_self_loops")=true,
            py::arg("with_parallel_edges")=false,
            py::arg("edge_proposer_type")="uniform",
            py::arg("block_proposer_type")="uniform",
            py::arg("sample_label_count_prob")=0.1,
            py::arg("label_creation_prob")=0.5,
            py::arg("shift")=1
        )
        ;
}

}

#endif
