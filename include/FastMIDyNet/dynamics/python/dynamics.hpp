#ifndef FAST_MIDYNET_PYTHON_DYNAMICS_HPP
#define FAST_MIDYNET_PYTHON_DYNAMICS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #include "FastMIDyNet/types.h"
// #include "FastMIDyNet/random_graph/random_graph.h"
// #include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = Dynamics>
class PyDynamics: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */

    /* Abstract methods */
};

}

#endif
