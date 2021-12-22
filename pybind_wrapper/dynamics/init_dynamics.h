#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_BASECLASS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/dynamics/python/dynamics.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

void initDynamicsBaseClass(py::module& m){

}

}
#endif
