#ifndef FAST_MIDYNET_PYTHON_PRIOR_HPP
#define FAST_MIDYNET_PYTHON_PRIOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename StateType, typename BaseClass = Prior<StateType>>
class PyPrior: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */
    void sampleState() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleState, ); }
    void samplePriors() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, samplePriors, ); }
    double getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogLikelihood, ); }
    double getLogPrior() const override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogPrior, ); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE_PURE(void, BaseClass, checkSelfConsistency, ); }

    /* Abstract methods */
    ~PyPrior() override = default;
    void setState(const StateType& state) override { PYBIND11_OVERRIDE(void, BaseClass, setState, state); }
    void isRoot(bool condition) override { PYBIND11_OVERRIDE(void, BaseClass, isRoot, condition); }
    void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }

};

}

#endif
