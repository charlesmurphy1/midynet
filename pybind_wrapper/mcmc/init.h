#ifndef FAST_MIDYNET_PYWRAPPER_INIT_MCMC_H
#define FAST_MIDYNET_PYWRAPPER_INIT_MCMC_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_mcmc.h"
#include "init_callbacks.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initMCMC(py::module& m){
    initCallBacks(m);
    initMCMCBaseClass(m);
}

}

#endif
