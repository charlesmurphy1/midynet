#ifndef FAST_MIDYNET_PYTHON_SBM_HPP
#define FAST_MIDYNET_PYTHON_SBM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/dcsbm.h"
#include "FastMIDyNet/random_graph/python/randomgraph.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{
//
template<typename BaseClass = StochasticBlockModelFamily>
class PyStochasticBlockModelFamily: public PyVertexLabeledRandomGraph<BlockIndex, BaseClass>{
protected:
    void _applyGraphMove (const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void _applyLabelMove (const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, _applyLabelMove, move); }
public:
    using PyVertexLabeledRandomGraph<BlockIndex, BaseClass>::PyVertexLabeledRandomGraph;

    /* Pure abstract methods */

    /* Abstract methods */
    void setBlockPrior(BlockPrior& blockPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setBlockPrior, blockPrior); }
    void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setEdgeMatrixPrior, edgeMatrixPrior); }

    const double getLogLikelihood() const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihood, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPrior, ); }
    const double getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioEdgeTerm, move); }
    const double getLogLikelihoodRatioAdjTerm (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioAdjTerm, move); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const double getLogLikelihoodRatioFromLabelMove (const BlockMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPriorRatioFromGraphMove, move); }
    const double getLogPriorRatioFromLabelMove (const BlockMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPriorRatioFromLabelMove, move); }
    void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
    void checkSelfSafety() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfSafety, ); }
    const bool isCompatible(const MultiGraph& graph) const override { PYBIND11_OVERRIDE(bool, BaseClass, isCompatible, graph); }
    bool isSafe() const override { PYBIND11_OVERRIDE(bool, BaseClass, isSafe, ); }
    // void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
    // void checkSelfSafety() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfSafety, ); }
    // void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }
};

}

#endif
