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
class PyBlockPrior: public PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, BaseClass> {
protected:
    void setBlockCountFromPartition(const BlockSequence& blocks) override { PYBIND11_OVERRIDE(void, BaseClass, setBlockCountFromPartition, blocks); }
public:
    using PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, BaseClass>::PyVertexLabeledPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }

    /* Overloaded abstract methods */
    void setState(const BlockSequence& blocks) override { PYBIND11_OVERRIDE(void, BaseClass, setState, blocks); }
};

}

#endif
