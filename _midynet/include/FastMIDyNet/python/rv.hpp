
#ifndef FAST_MIDYNET_PYTHON_RV_HPP
#define FAST_MIDYNET_PYTHON_RV_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/rv.hpp"


namespace py = pybind11;
namespace FastMIDyNet{

template<typename BaseClass = NestedRandomVariable>
class PyNestedRandomVariable: public BaseClass{
public:
    using BaseClass::BaseClass;

    /* Pure abstract methods */

    /* Abstract & overloaded methods */
    bool isRoot(bool condition) const override { PYBIND11_OVERRIDE(bool, BaseClass, isRoot, condition);}
    bool isProcessed(bool condition) const override { PYBIND11_OVERRIDE(bool, BaseClass, isProcessed, condition);}
    void checkSelfConsistency() const override  { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, );}
    void checkSelfSafety() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfSafety, );}
    void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, );}
    bool isSafe() const override { PYBIND11_OVERRIDE(bool, BaseClass, isSafe, );}
};


}

#endif
