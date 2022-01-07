#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RNG_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RNG_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/rng.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initRNG(py::module& m){
    m.def("seed", &seed, py::arg("n"));
    m.def("seedWithTime", &seedWithTime);
}

}

#endif
