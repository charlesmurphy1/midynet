#ifndef FAST_MIDYNET_PYTHON_RV_HPP
#define FAST_MIDYNET_PYTHON_RV_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = NestedRandomVariable>
class PyNestedRandomVariable: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */

    /* Abstract methods */
    bool isRoot(bool condition) const override { PYBIND11_OVERRIDE(bool, BaseClass, isRoot, condition); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
    void checkSafety() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSafety, ); }
    void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }

};

}

#endif
