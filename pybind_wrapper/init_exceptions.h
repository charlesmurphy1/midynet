#ifndef FAST_MIDYNET_PYWRAPPER_INIT_EXCEPTIONS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_EXCEPTIONS_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "FastMIDyNet/exceptions.h"

void initExceptions(pybind11::module& m){
    m.def("assert_valid_probability", &FastMIDyNet::assertValidProbability);
    pybind11::class_<FastMIDyNet::ConsistencyError>(m, "ConsistencyError")
        .def(pybind11::init<std::string>(), pybind11::arg("message")="");
}

#endif
