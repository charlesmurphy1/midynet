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

    py::class_<CallBackList>(m, "CallBackList")
        .def(py::init<>())
        .def(py::init<std::vector<CallBack*>>(), py::arg("callbacks"))
        .def(py::init<const CallBackList&>(), py::arg("callbackList"))
        .def("set_up", &CallBackList::setUp, py::arg("mcmc"))
        .def("tear_down", &CallBackList::tearDown)
        .def("on_begin", &CallBackList::onBegin)
        .def("on_end", &CallBackList::onEnd)
        .def("on_step_begin", &CallBackList::onStepBegin)
        .def("on_step_end", &CallBackList::onStepEnd)
        .def("on_sweep_begin", &CallBackList::onSweepBegin)
        .def("on_sweep_end", &CallBackList::onSweepEnd)
        .def("push_back", &CallBackList::pushBack)
        .def("pop_back", &CallBackList::popBack)
        .def("remove", &CallBackList::remove) ;
    initVerbose(m);
    initCollectors(m);
}

}

#endif
