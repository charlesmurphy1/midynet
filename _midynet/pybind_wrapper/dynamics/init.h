#ifndef FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_DYNAMICS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_dynamics.h"

namespace py = pybind11;
namespace FastMIDyNet{

void initDynamics(py::module& m){
    declareDynamicsBaseClass<RandomGraph>(m, "Dynamics");
    declareDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDynamics");

    declareBinaryDynamicsBaseClass<RandomGraph>(m, "BinaryDynamics");
    declareBinaryDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledBinaryDynamics");

    declareCowanDynamicsBaseClass<RandomGraph>(m, "CowanDynamics");
    declareCowanDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledCowanDynamics");

    declareDegreeDynamicsBaseClass<RandomGraph>(m, "DegreeDynamics");
    declareDegreeDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDegreeDynamics");

    declareGlauberDynamicsBaseClass<RandomGraph>(m, "GlauberDynamics");
    declareGlauberDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledGlauberDynamics");

    declareSISDynamicsBaseClass<RandomGraph>(m, "SISDynamics");
    declareSISDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledSISDynamics");



}

}

#endif
