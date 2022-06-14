#ifndef FAST_MIDYNET_PYTHON_BLOCKCOUNT_H
#define FAST_MIDYNET_PYTHON_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"



namespace FastMIDyNet{

template <typename BaseClass = BlockCountPrior>
class PyBlockCountPrior: public PyPrior<size_t, BaseClass> {
public:
    using PyPrior<size_t, BaseClass>::PyPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodFromState(const size_t& state) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodFromState, state); }

    /* Overloaded abstract methods */
    void samplePriors() override { PYBIND11_OVERRIDE(void, BaseClass, samplePriors, ); }
    const double getLogLikelihood() const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihood, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPrior, ); }
};

}

#endif
