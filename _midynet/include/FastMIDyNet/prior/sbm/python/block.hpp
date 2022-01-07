#ifndef FAST_MIDYNET_PYTHON_BLOCK_H
#define FAST_MIDYNET_PYTHON_BLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/python/prior.hpp"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"
#include "FastMIDyNet/prior/sbm/block.h"


namespace FastMIDyNet{

template <typename BlockPriorBaseClass = BlockPrior>
class PyBlockPrior: public PyPrior<std::vector<size_t>, BlockPriorBaseClass> {
public:
    using PyPrior<std::vector<size_t>, BlockPriorBaseClass>::PyPrior;
    /* Pure abstract methods */
    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(double, BlockPriorBaseClass, getLogLikelihoodRatioFromBlockMove, move); }
    double getLogPriorRatioFromBlockMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(double, BlockPriorBaseClass, getLogPriorRatioFromBlockMove, move); }
    void applyBlockMove(const BlockMove& move) override { PYBIND11_OVERRIDE_PURE(void, BlockPriorBaseClass, applyBlockMove, move); }

    /* Overloaded abstract methods */
    ~PyBlockPrior() override = default;
    const size_t& getBlockCount() const override { PYBIND11_OVERRIDE(const size_t&, BlockPriorBaseClass, getBlockCount, ); }
    const std::vector<size_t>& getVertexCountsInBlocks() const override { PYBIND11_OVERRIDE(const std::vector<size_t>&, BlockPriorBaseClass, getVertexCountsInBlocks, ); }
};

}

#endif
