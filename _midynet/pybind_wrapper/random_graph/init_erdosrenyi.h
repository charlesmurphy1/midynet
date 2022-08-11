#ifndef FAST_MIDYNET_PYWRAPPER_INIT_ERDOSRENYI_H
#define FAST_MIDYNET_PYWRAPPER_INIT_ERDOSRENYI_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/random_graph/erdosrenyi.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initErdosRenyi(py::module& m){
    py::class_<ErdosRenyiModelBase, RandomGraph>(m, "ErdosRenyiModelBase")
        // .def(py::init<size_t, bool, bool>(), py::arg("size"), py::arg("with_self_loops")=true, py::arg("with_parallel_edges")=true)
        // .def(py::init<size_t, EdgeCountPrior&, bool, bool>(), py::arg("size"), py::arg("edge_count_prior"), py::arg("with_self_loops")=true, py::arg("with_parallel_edges")=true)
        .def("get_edge_count_prior", &ErdosRenyiModelBase::getEdgeCountPrior)
        .def("set_edge_count_prior", &ErdosRenyiModelBase::setEdgeCountPrior, py::arg("prior"))
        ;
    py::class_<ErdosRenyiModel, ErdosRenyiModelBase>(m, "ErdosRenyiModel")
        .def(
            py::init<size_t, double, bool, bool, bool, std::string>(),
            py::arg("size"), py::arg("edge_count"),
            py::arg("canonical")=false,
            py::arg("with_self_loops")=true,
            py::arg("with_parallel_edges")=true,
            py::arg("edge_proposer_type")="uniform"
        )
        ;

}

}

#endif
