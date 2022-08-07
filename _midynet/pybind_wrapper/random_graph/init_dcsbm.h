#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DCSBM_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/dcsbm.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initDegreeCorrectedStochasticBlockModel(py::module& m){
    py::class_<DegreeCorrectedStochasticBlockModelBase, BlockLabeledRandomGraph>(m, "DegreeCorrectedStochasticBlockModelBase")
        .def(
            py::init<size_t, VertexLabeledDegreePrior&>(), py::arg("size"), py::arg("degree_prior")
        )
        .def("get_degree_prior", &DegreeCorrectedStochasticBlockModelBase::getDegreePrior)
        .def("set_degree_prior", &DegreeCorrectedStochasticBlockModelBase::setDegreePrior, py::arg("prior"))
        ;

    py::class_<DegreeCorrectedStochasticBlockModelFamily, DegreeCorrectedStochasticBlockModelBase>(m, "DegreeCorrectedStochasticBlockModelFamily")
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
