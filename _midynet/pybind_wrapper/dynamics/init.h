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
    declareDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDynamics");

    declareBinaryDynamicsBaseClass<RandomGraph>(m, "BinaryDynamics");
    declareBinaryDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledBinaryDynamics");
    declareBinaryDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledBinaryDynamics");

    declareCowanDynamicsBaseClass<RandomGraph>(m, "CowanDynamics");
    declareCowanDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledCowanDynamics");
    declareCowanDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledCowanDynamics");

    declareDegreeDynamicsBaseClass<RandomGraph>(m, "DegreeDynamics");
    declareDegreeDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDegreeDynamics");
    declareDegreeDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDegreeDynamics");

    declareGlauberDynamicsBaseClass<RandomGraph>(m, "GlauberDynamics");
    declareGlauberDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledGlauberDynamics");
    declareGlauberDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledGlauberDynamics");

    declareSISDynamicsBaseClass<RandomGraph>(m, "SISDynamics");
    declareSISDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledSISDynamics");
    declareSISDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledSISDynamics");



}

}

#endif
