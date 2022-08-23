#ifndef FAST_MIDYNET_PYTHON_BLOCKCOUNT_H
#define FAST_MIDYNET_PYTHON_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/prior/prior.hpp"
#include "FastMIDyNet/random_graph/prior/python/prior.hpp"
#include "FastMIDyNet/random_graph/prior/block_count.h"



namespace FastMIDyNet{

template <typename BaseClass = BlockCountPrior>
class PyBlockCountPrior: public PyVertexLabeledPrior<size_t, BlockIndex, BaseClass> {
public:
    using PyVertexLabeledPrior<size_t, BlockIndex, BaseClass>::PyVertexLabeledPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodFromState(const size_t& state) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodFromState, state); }
};

template <typename BaseClass = NestedBlockCountPrior>
class PyNestedBlockCountPrior: public PyBlockCountPrior<BaseClass> {
public:
    using PyBlockCountPrior<BaseClass>::PyBlockCountPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodFromNestedState(const std::vector<size_t>& state) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodFromState, state); }
};

}

#endif
