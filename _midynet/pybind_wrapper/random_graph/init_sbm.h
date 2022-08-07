#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/sbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initStochasticBlockModel(py::module& m){
    py::class_<StochasticBlockModelBase, BlockLabeledRandomGraph>(m, "StochasticBlockModelBase")
        .def(
            py::init<size_t, LabelGraphPrior&, bool, bool, bool>(),
            py::arg("size"),
            py::arg("label_graph_prior"),
            py::arg("stub_labeled")=true,
            py::arg("with_self_loops")=true,
            py::arg("with_parallel_edges")=true
        )
        .def("get_label_graph_prior", &StochasticBlockModelBase::getLabelGraphPrior)
        .def("set_label_graph_prior", &StochasticBlockModelBase::setLabelGraphPrior, py::arg("prior"))
        ;

    py::class_<StochasticBlockModel, StochasticBlockModelBase>(m, "StochasticBlockModel")
        .def(
            py::init<const std::vector<BlockIndex>, const LabelGraph&, bool, bool, bool>(),
            py::arg("blocks"),
            py::arg("label_graph"),
            py::arg("stub_labeled")=true,
            py::arg("with_self_loops")=true,
            py::arg("with_parallel_edges")=true
        )
        ;

    py::class_<StochasticBlockModelFamily, StochasticBlockModelBase>(m, "StochasticBlockModelFamily")
        .def(
            py::init<size_t, double, bool, bool, bool, bool, bool>(),
            py::arg("size"),
            py::arg("edge_count"),
            py::arg("hyperprior")=true,
            py::arg("canonical")=false,
            py::arg("stub_labeled")=true,
            py::arg("with_self_loops")=true,
            py::arg("with_parallel_edges")=true
        )
        ;
}

}

#endif
