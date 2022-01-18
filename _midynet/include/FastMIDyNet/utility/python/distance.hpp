#ifndef FAST_MIDYNET_PYTHON_DISTANCE_HPP
#define FAST_MIDYNET_PYTHON_DISTANCE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility/distance.h"

namespace FastMIDyNet{

/* CallBack  base class */
template<typename BaseClass = GraphDistance>
class PyGraphDistance: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */
    double compute(const MultiGraph& g1, const MultiGraph& g2) const override { PYBIND11_OVERRIDE(double, BaseClass, compute, g1, g2); }

    /* Abstract methods */
};

}

#endif
