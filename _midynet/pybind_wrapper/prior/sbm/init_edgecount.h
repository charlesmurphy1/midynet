#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGECOUNT_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_EDGECOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "declare.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/python/edgecount.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initEdgeCountPrior(py::module& m){
    declareSBMPrior<EdgeCountPrior, Prior<size_t>, PyEdgeCountPrior<>>(m, "EdgeCountPrior");

    py::class_<EdgeCountDeltaPrior, EdgeCountPrior>(m, "EdgeCountDeltaPrior")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("edge_count"));

    py::class_<EdgeCountPoissonPrior, EdgeCountPrior>(m, "EdgeCountPoissonPrior")
        .def(py::init<>())
        .def(py::init<double>(), py::arg("mean"));
}

}

#endif
