#ifndef FAST_MIDYNET_PYTHON_MCMC_HPP
#define FAST_MIDYNET_PYTHON_MCMC_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/mcmc/mcmc.h"

namespace FastMIDyNet{

template<typename BaseClass = MCMC>
class PyMCMC: public PyNestedRandomVariable<BaseClass>{
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
    /* Pure abstract methods */
    void sample() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sample, ); }
    void samplePrior() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, samplePrior, ); }
    const double getLogLikelihood() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, );
    }
    const double getLogPrior() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPrior, );
    }
    const double getLogJoint() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogJoint, );
    }
    bool doMetropolisHastingsStep() override {
        PYBIND11_OVERRIDE_PURE(bool, BaseClass, doMetropolisHastingsStep, );
    }

    /* Abstract methods */
    void setUp() override { PYBIND11_OVERRIDE(void, BaseClass, setUp, ); }
    void tearDown() override { PYBIND11_OVERRIDE(void, BaseClass, tearDown, ); }
    void removeCallBack(std::string key) override {
        PYBIND11_OVERRIDE(void, BaseClass, removeCallBack, key);
    }
    void onStepBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onStepBegin, ); }
    void onStepEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onStepEnd, ); }
    void onSweepBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onSweepBegin, ); }
    void onSweepEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onSweepEnd, ); }


};



}

#endif
