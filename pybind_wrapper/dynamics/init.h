#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_dynamics.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initDynamics(py::module& m){
    initDynamicsBaseClass(m);
}

}

#endif
