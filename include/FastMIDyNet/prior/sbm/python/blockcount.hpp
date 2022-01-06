#ifndef FAST_MIDYNET_PYTHON_BLOCKCOUNT_H
#define FAST_MIDYNET_PYTHON_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"



namespace FastMIDyNet{

template <typename BlockCountPriorBaseClass = BlockCountPrior>
class PyBlockCountPrior: public PyPrior<size_t, BlockCountPriorBaseClass> {
public:
    using PyPrior<size_t, BlockCountPriorBaseClass>::PyPrior;
    /* Pure abstract methods */
    double getLogLikelihoodFromState(const size_t& state) const override { PYBIND11_OVERRIDE_PURE(double, BlockCountPriorBaseClass, getLogLikelihoodFromState, state); }

    /* Overloaded abstract methods */
    void samplePriors() override { PYBIND11_OVERRIDE(void, BlockCountPriorBaseClass, samplePriors, ); }
    double getLogLikelihood() const override { PYBIND11_OVERRIDE(double, BlockCountPriorBaseClass, getLogLikelihood, ); }
    double getLogPrior() const override { PYBIND11_OVERRIDE(double, BlockCountPriorBaseClass, getLogPrior, ); }
};

}

#endif
