#ifndef FAST_MIDYNET_PYTHON_LABELGRAPH_H
#define FAST_MIDYNET_PYTHON_LABELGRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/prior/python/prior.hpp"
#include "FastMIDyNet/random_graph/prior/prior.hpp"
#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"


namespace FastMIDyNet{

template <typename BaseClass = LabelGraphPrior>
class PyLabelGraphPrior: public PyVertexLabeledPrior< LabelGraph, BlockIndex, BaseClass> {
protected:
    void destroyBlock(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, destroyBlock, move); }
    void applyGraphMoveToState(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMoveToState, move); }
    void applyLabelMoveToState(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyLabelMoveToState, move); }
    void recomputeStateFromGraph() override { PYBIND11_OVERRIDE(void, BaseClass, recomputeStateFromGraph, ); }
    void recomputeConsistentState() override { PYBIND11_OVERRIDE(void, BaseClass, recomputeConsistentState, ); }
public:
    using PyVertexLabeledPrior<LabelGraph, BlockIndex, BaseClass>::PyVertexLabeledPrior;
    /* Pure abstract methods */
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }

    /* Overloaded abstract methods */
    void setPartition(const std::vector<BlockIndex>& partition) override { PYBIND11_OVERRIDE(void, BaseClass, setPartition, partition); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
};

}

#endif
