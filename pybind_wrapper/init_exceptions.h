#ifndef FAST_MIDYNET_PYWRAPPER_INIT_EXCEPTIONS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_EXCEPTIONS_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "FastMIDyNet/exceptions.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initExceptions(py::module& m){
    m.def("assert_valid_probability", &assertValidProbability);
    py::class_<ConsistencyError>(m, "ConsistencyError")
        .def(py::init<std::string>(), py::arg("message")="");
}

}

#endif
