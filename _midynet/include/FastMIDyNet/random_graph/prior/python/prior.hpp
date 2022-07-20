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
    using cdouble = const double;
    ~PyPrior() override = default;
    /* Pure abstract methods */
    void sampleState() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleState, ); }
    void samplePriors() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, samplePriors, ); }
    void setState(const StateType& state) override { PYBIND11_OVERRIDE(void, BaseClass, setState, state); }
    cdouble getLogLikelihood() const override  { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, getLogLikelihood, ); }
    cdouble getLogPrior() const override { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, getLogPrior, ); }

    /* Abstract methods */
protected:
    void _applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _applyGraphMove, move); }
    cdouble _getLogJointRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, _getLogJointRatioFromGraphMove, move); }
};


template<typename StateType, typename Label, typename BaseClass = VertexLabeledPrior<StateType, Label>>
class PyVertexLabeledPrior: public PyPrior<StateType, BaseClass>{
public:
    using PyPrior<StateType, BaseClass>::PyPrior;
    ~PyVertexLabeledPrior() override = default;

    /* Pure abstract methods */
protected:
    void _applyLabelMove(const LabelMove<Label>& move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _applyLabelMove, move); }
    const double _getLogJointRatioFromLabelMove(const LabelMove<Label>& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, _getLogJointRatioFromLabelMove, move); }

    /* Abstract methods */
};

}

#endif
