#ifndef FAST_MIDYNET_PYWRAPPER_INIT_UTILITY_H
#define FAST_MIDYNET_PYWRAPPER_INIT_UTILITY_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/utility/maps.hpp"
#include "init_maps.h"
// #include "init_distance.h"

namespace py = pybind11;
namespace FastMIDyNet{


void initUtility(py::module& m){
    initMaps(m);
    // initDistances(m);
}

}

#endif
