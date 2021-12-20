#ifndef FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_H
#define FAST_MIDYNET_PYWRAPPER_INIT_SBMPRIOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_edgecount.h"



void initSBMPrior(pybind11::module& m){
    initEdgeCountPrior(m);
}

#endif
