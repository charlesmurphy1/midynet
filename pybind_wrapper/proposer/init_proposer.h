#ifndef FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_H
#define FAST_MIDYNET_PYWRAPPER_INIT_PROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_movetypes.h"

void initProposer(pybind11::module& m){
    initMoveTypes(m);
}

#endif
