#ifndef FAST_MIDYNET_PYWRAPPER_INIT_HDCSBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_HDCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/hdcsbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initNestedDegreeCorrectedStochasticBlockModel(py::module& m){
    py::class_<NestedDegreeCorrectedStochasticBlockModelBase, NestedBlockLabeledRandomGraph>(m, "NestedDegreeCorrectedStochasticBlockModelBase")
        // .def(
        //     py::init<size_t, EdgeCountPrior&, VertexLabeledDegreePrior&>(),
        //     py::arg("size"),
        //     py::arg("edge_count_prior"),
        //     py::arg("degree_prior")
        // )
        .def("get_degree_prior", &NestedDegreeCorrectedStochasticBlockModelBase::getDegreePrior)
        .def("set_degree_prior", &NestedDegreeCorrectedStochasticBlockModelBase::setDegreePrior, py::arg("prior"))
        ;

    py::class_<NestedDegreeCorrectedStochasticBlockModelFamily, NestedDegreeCorrectedStochasticBlockModelBase>(m, "NestedDegreeCorrectedStochasticBlockModelFamily")
        .def(
            py::init<size_t, double, bool, bool, std::string, std::string, double, double, double>(),
            py::arg("size"),
            py::arg("edge_count"),
            py::arg("hyperprior")=true,
            py::arg("canonical")=false,
            py::arg("edge_proposer_type")="uniform",
            py::arg("block_proposer_type")="uniform",
            py::arg("sample_label_prob")=0.1,
            py::arg("label_creation_prob")=0.5,
            py::arg("shift")=1
        )
        ;
}

}

#endif
