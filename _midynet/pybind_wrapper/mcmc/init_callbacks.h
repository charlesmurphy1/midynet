#ifndef FAST_MIDYNET_PYWRAPPER_INIT_CALLBACKS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_CALLBACKS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_verbose.h"
#include "init_collectors.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initCallBacks(py::module& m){
    py::class_<CallBack, PyCallBack<>>(m, "CallBack")
        .def(py::init<>())
        .def("set_up", &CallBack::setUp, py::arg("mcmc"))
        .def("tear_down", &CallBack::tearDown)
        .def("on_begin", &CallBack::onBegin)
        .def("on_end", &CallBack::onEnd)
        .def("on_step_begin", &CallBack::onStepBegin)
        .def("on_step_end", &CallBack::onStepEnd)
        .def("on_sweep_begin", &CallBack::onSweepBegin)
        .def("on_sweep_end", &CallBack::onSweepEnd) ;

    py::class_<CallBackMap>(m, "CallBackDict")
        .def(py::init<>())
        .def("set_up", [&](CallBackMap& self, MCMC& mcmc){ self.setUp(&mcmc); })
        .def("tear_down", &CallBackMap::tearDown)
        .def("on_begin", &CallBackMap::onBegin)
        .def("on_end", &CallBackMap::onEnd)
        .def("on_step_begin", &CallBackMap::onStepBegin)
        .def("on_step_end", &CallBackMap::onStepEnd)
        .def("on_sweep_begin", &CallBackMap::onSweepBegin)
        .def("on_sweep_end", &CallBackMap::onSweepEnd)
        .def("insert", [&](CallBackMap& self, std::string key, CallBack& value) {
            self.insert(key, value); }, py::arg("key"), py::arg("callback"))
        .def("remove", &CallBackMap::remove, py::arg("key"))
        .def("size", &CallBackMap::size) ;
    initVerbose(m);
    initCollectors(m);
}

}

#endif
