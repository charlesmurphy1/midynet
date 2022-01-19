#ifndef FAST_MIDYNET_PYWRAPPER_INIT_UTILITY_FUNCTIONS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_UTILITY_FUNCTIONS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/utility/functions.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initFunctions(py::module& m){
    m.def("log_factorial", &logFactorial, py::arg("n"));
    m.def("log_double_factorial", &logDoubleFactorial, py::arg("n"));
    m.def("log_binom", &logBinomialCoefficient, py::arg("n"), py::arg("k"));
    m.def("log_poisson", &logPoissonPMF, py::arg("k"), py::arg("mean"));
    m.def("log_truncpoisson", &logZeroTruncatedPoissonPMF, py::arg("k"), py::arg("mean"));
    m.def("log_multinom", py::overload_cast<std::list<size_t>>(&logMultinomialCoefficient), py::arg("kList"));
    m.def("log_multinom", py::overload_cast<std::vector<size_t>>(&logMultinomialCoefficient), py::arg("kVec"));
    m.def("log_multiset", &logMultisetCoefficient, py::arg("n"), py::arg("k"));
}

}

#endif
