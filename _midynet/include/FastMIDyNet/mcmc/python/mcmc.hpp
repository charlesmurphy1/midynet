#ifndef FAST_MIDYNET_PYTHON_MCMC_HPP
#define FAST_MIDYNET_PYTHON_MCMC_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/mcmc/mcmc.h"

namespace FastMIDyNet{

template<typename BaseClass = MCMC>
class PyMCMC: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */
    const MultiGraph& getGraph() const override {
        PYBIND11_OVERRIDE_PURE(const MultiGraph&, BaseClass, getGraph, );
    }
    const std::vector<BlockIndex>& getBlocks() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<BlockIndex>&, BaseClass, getBlocks, );
    }
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
    void checkSafety() const { PYBIND11_OVERRIDE(void, BaseClass, checkSafety, ); }

};




}

#endif
