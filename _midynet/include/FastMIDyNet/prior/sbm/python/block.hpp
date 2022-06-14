#ifndef FAST_MIDYNET_PYTHON_BLOCK_H
#define FAST_MIDYNET_PYTHON_BLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"


namespace FastMIDyNet{

template <typename BaseClass = BlockPrior>
class PyBlockPrior: public PyPrior<std::vector<size_t>, BaseClass> {
public:
    using PyPrior<std::vector<size_t>, BaseClass>::PyPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromBlockMove, move); }

    /* Overloaded abstract methods */
    void setState(const BlockSequence& blocks) override { PYBIND11_OVERRIDE(void, BaseClass, setState, blocks); }
};

}

#endif
