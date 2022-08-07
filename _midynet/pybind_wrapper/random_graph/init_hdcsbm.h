#ifndef FAST_MIDYNET_PYWRAPPER_INIT_HDCSBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_HDCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/hdcsbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initNestedDegreeCorrectedStochasticBlockModel(py::module& m){
    py::class_<NestedDegreeCorrectedStochasticBlockModelBase, NestedBlockLabeledRandomGraph>(m, "NestedDegreeCorrectedStochasticBlockModelBase")
        .def(
            py::init<size_t, EdgeCountPrior&, VertexLabeledDegreePrior&>(),
            py::arg("size"),
            py::arg("edge_count_prior"),
            py::arg("degree_prior")
        )
        .def("get_degree_prior", &NestedDegreeCorrectedStochasticBlockModelBase::getDegreePrior)
        .def("set_degree_prior", &NestedDegreeCorrectedStochasticBlockModelBase::setDegreePrior, py::arg("prior"))
        ;

    py::class_<NestedDegreeCorrectedStochasticBlockModelFamily, NestedDegreeCorrectedStochasticBlockModelBase>(m, "NestedDegreeCorrectedStochasticBlockModelFamily")
        .def(
            py::init<size_t, double, bool, bool>(),
            py::arg("size"),
            py::arg("edge_count"),
            py::arg("hyperprior")=true,
            py::arg("canonical")=false
        )
        ;
}

}

#endif
