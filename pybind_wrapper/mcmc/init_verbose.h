#ifndef FAST_MIDYNET_PYWRAPPER_INIT_VERBOSE_H
#define FAST_MIDYNET_PYWRAPPER_INIT_VERBOSE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/mcmc/callbacks/verbose.h"
#include "FastMIDyNet/mcmc/python/callback.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initVerbose(py::module& m){
    /* Verbose classes */
    py::class_<Verbose, CallBack, PyVerbose<>>(m, "Verbose")
        .def(py::init<>())
        .def("get_name", &Verbose::getName)
        .def("get_message", &Verbose::getMessage);

    py::class_<TimerVerbose, Verbose>(m, "TimerVerbose")
        .def(py::init<>());

    py::class_<SuccessCounterVerbose, Verbose>(m, "SuccessCounterVerbose")
        .def(py::init<>());

    py::class_<FailureCounterVerbose, Verbose>(m, "FailureCounterVerbose")
        .def(py::init<>());

    /* LogJointRatioVerbose classes */
    py::class_<LogJointRatioVerbose, PyLogJointRatioVerbose<>>(m, "LogJointRatioVerbose")
        .def(py::init<>())
        .def("update_saved_ratio", &LogJointRatioVerbose::updateSavedRatio);

    py::class_<MinimumLogJointRatioVerbose, Verbose>(m, "MinimumLogJointRatioVerbose")
        .def(py::init<>());

    py::class_<MaximumLogJointRatioVerbose, Verbose>(m, "MaximumLogJointRatioVerbose")
        .def(py::init<>());

    py::class_<MeanLogJointRatioVerbose, Verbose>(m, "MeanLogJointRatioVerbose")
        .def(py::init<>());

    /* VerboseDisplay classes */
    py::class_<VerboseDisplay, CallBack, PyVerboseDisplay<>>(m, "VerboseDisplay")
        .def(py::init<std::vector<Verbose*>>(), py::arg("verboses")={})
        .def("get_message", &VerboseDisplay::getMessage)
        .def("write_message", py::overload_cast<std::string>(&VerboseDisplay::writeMessage),
            py::arg("message"))
        .def("write_message", py::overload_cast<>(&VerboseDisplay::writeMessage));

    py::class_<VerboseToConsole, VerboseDisplay>(m, "VerboseToConsole")
        .def(py::init<std::vector<Verbose*>>(), py::arg("verboses")={}) ;

    py::class_<VerboseToFile, VerboseDisplay>(m, "VerboseToFile")
        .def(py::init<std::string, std::vector<Verbose*>>(),
            py::arg("filename")="verbose", py::arg("verboses") = {}) ;
}

}

#endif
