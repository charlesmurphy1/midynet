#ifndef FAST_MIDYNET_PYWRAPPER_INIT_CALLBACKS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_CALLBACKS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_verbose.h"
#include "init_actions.h"
#include "init_collectors.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename MCMCType>
py::class_<CallBack<MCMCType>, PyCallBack<MCMCType>> declareCallBack(py::module& m, std::string pyName){
    return py::class_<CallBack<MCMCType>, PyCallBack<MCMCType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("set_up", &CallBack<MCMCType>::setUp, py::arg("mcmc"))
        .def("tear_down", &CallBack<MCMCType>::tearDown)
        .def("on_begin", &CallBack<MCMCType>::onBegin)
        .def("on_end", &CallBack<MCMCType>::onEnd)
        .def("on_step_begin", &CallBack<MCMCType>::onStepBegin)
        .def("on_step_end", &CallBack<MCMCType>::onStepEnd)
        .def("on_sweep_begin", &CallBack<MCMCType>::onSweepBegin)
        .def("on_sweep_end", &CallBack<MCMCType>::onSweepEnd)
        .def("clear", &CallBack<MCMCType>::clear)
        ;
}

template<typename MCMCType>
py::class_<CallBackMap<MCMCType>> declareCallBackMap(py::module& m, std::string pyName){
    return py::class_<CallBackMap<MCMCType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("set_up", [](CallBackMap<MCMCType>& self, MCMC& mcmc){ self.setUp(&mcmc); })
        .def("tear_down", &CallBackMap<MCMCType>::tearDown)
        .def("on_begin", &CallBackMap<MCMCType>::onBegin)
        .def("on_end", &CallBackMap<MCMCType>::onEnd)
        .def("on_step_begin", &CallBackMap<MCMCType>::onStepBegin)
        .def("on_step_end", &CallBackMap<MCMCType>::onStepEnd)
        .def("on_sweep_begin", &CallBackMap<MCMCType>::onSweepBegin)
        .def("on_sweep_end", &CallBackMap<MCMCType>::onSweepEnd)
        .def("get", [](CallBackMap<MCMCType>& self, std::string key) { return self.get(key); }, py::arg("key"))
        .def("insert", [](CallBackMap<MCMCType>& self, std::string key, CallBack<MCMCType>& value) {
            self.insert(key, value); }, py::arg("key"), py::arg("callback"))
        .def("remove", &CallBackMap<MCMCType>::remove, py::arg("key"))
        .def("size", &CallBackMap<MCMCType>::size)
        .def("contains", &CallBackMap<MCMCType>::contains, py::arg("key"))
        .def("clear", &CallBackMap<MCMCType>::clear)
        .def("__contains__", [](CallBackMap<MCMCType> self, std::string key){ return self.contains(key); }, py::arg("key"))
        .def("__getitem__", [](CallBackMap<MCMCType> self, std::string key){ return self.get(key); }, py::arg("key"))
        .def("__delitem__", [](CallBackMap<MCMCType> self, std::string key){ return self.remove(key); }, py::arg("key"))
        ;
}

void initCallBacks(py::module& m){
    declareCallBack<MCMC>(m, "CallBack");
    declareCallBack<VertexLabelMCMC<BlockIndex>>(m, "BlockCallBack");
    declareCallBack<GraphReconstructionMCMC<RandomGraph>>(m, "GraphReconstructionCallBack");
    declareCallBack<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>(m, "BlockGraphReconstructionCallBack");
    initVerbose(m);
    initActions(m);
    initCollectors(m);
}

}

#endif
