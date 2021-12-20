#ifndef FAST_MIDYNET_PYWRAPPER_INIT_RNG_H
#define FAST_MIDYNET_PYWRAPPER_INIT_RNG_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/rng.h"

void initRNG(pybind11::module& m){
    m.def("seed", &FastMIDyNet::seed, pybind11::arg("n"));
    m.def("seedWithTime", &FastMIDyNet::seedWithTime);
}

#endif
