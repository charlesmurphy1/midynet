#ifndef FAST_MIDYNET_PYTHON_PRIOR_HPP
#define FAST_MIDYNET_PYTHON_PRIOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename StateType, typename BaseClass = Prior<StateType>>
class PyPrior: public PyNestedRandomVariable<BaseClass>{
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
    ~PyPrior() override = default;
    /* Pure abstract methods */
    void sampleState() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleState, ); }
    void samplePriors() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, samplePriors, ); }
    const double getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPrior, ); }

    /* Abstract methods */
    void setState(const StateType& state) override { PYBIND11_OVERRIDE(void, BaseClass, setState, state); }
protected:
    void onBlockCreation(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, onBlockCreation, move); }
    void onBlockDeletion(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, onBlockDeletion, move); }
    void _applyBlockMove(const BlockMove& move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _applyBlockMove, move); }
    void _applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _applyGraphMove, move); }
    const double _getLogJointRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, _getLogJointRatioFromGraphMove, move); }
    const double _getLogJointRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, _getLogJointRatioFromBlockMove, move); }

};

}

#endif
