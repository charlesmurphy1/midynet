#ifndef FAST_MIDYNET_PYWRAPPER_INIT_ACTIONS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_ACTIONS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/mcmc/callbacks/callback.hpp"
#include "FastMIDyNet/mcmc/callbacks/action.h"
#include "FastMIDyNet/mcmc/mcmc.h"
// #include "FastMIDyNet/utility/distance.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initActions(py::module&m){
    py::class_<CheckConsistencyOnSweep, CallBack<MCMC>>(m, "CheckConsistencyOnSweep")
        .def(py::init<>())
        ;

    py::class_<CheckSafetyOnSweep, CallBack<MCMC>>(m, "CheckSafetyOnSweep")
        .def(py::init<>())
        ;
}

}
#endif
