#ifndef FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_DEGREE_H
#define FAST_MIDYNET_PYWRAPPER_PRIOR_INIT_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/random_graph/prior/python/prior.hpp"
#include "FastMIDyNet/random_graph/prior/python/degree.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/degree.h"


namespace py = pybind11;
namespace FastMIDyNet{

void initDegreePrior(py::module& m){
    py::class_<DegreePrior, Prior<std::vector<size_t>>, PyDegreePrior<>>(m, "DegreePrior")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&>(), py::arg("size"), py::arg("edge_count_prior"))
        .def("get_size", &DegreePrior::getSize)
        .def("get_edge_count", &DegreePrior::getEdgeCount)
        .def("get_degree_of_idx", &DegreePrior::getDegreeOfIdx)
        .def("get_degree_counts", &DegreePrior::getDegreeCounts)
        .def("get_edge_count_prior", &DegreePrior::getEdgeCountPrior)
        .def("set_edge_count_prior", &DegreePrior::setEdgeCountPrior, py::arg("edge_count_prior"))
        ;

    py::class_<DegreeDeltaPrior, DegreePrior>(m, "DegreeDeltaPrior")
        .def(py::init<const DegreeSequence&>(), py::arg("degrees"))
        ;

    py::class_<DegreeUniformPrior, DegreePrior>(m, "DegreeUniformPrior")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&>(), py::arg("size"), py::arg("edge_count_prior"))
        ;

    py::class_<DegreeUniformHyperPrior, DegreePrior>(m, "DegreeUniformHyperPrior")
        .def(py::init<size_t>(), py::arg("size"))
        .def(py::init<size_t, EdgeCountPrior&>(), py::arg("size"), py::arg("edge_count_prior"))
        ;


}

}

#endif
