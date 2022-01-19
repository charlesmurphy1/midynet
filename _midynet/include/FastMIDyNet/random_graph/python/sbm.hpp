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
class PyStochasticBlockModelFamily: public PyRandomGraph<BaseClass>{
public:
    using PyRandomGraph<StochasticBlockModelFamily>::PyRandomGraph;

    /* Pure abstract methods */

    /* Abstract methods */
    void setBlockPrior(BlockPrior& blockPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setBlockPrior, blockPrior); }
    void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setEdgeMatrixPrior, edgeMatrixPrior); }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const {
        PYBIND11_OVERRIDE(const std::vector<CounterMap<size_t>>&, BaseClass, getDegreeCountsInBlocks, );
    }
    const std::vector<size_t>& getDegrees() const {
        PYBIND11_OVERRIDE(const std::vector<size_t>&, BaseClass, getDegrees, );
    }
    const double getLogLikelihood() const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihood, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPrior, ); }
    const double getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioEdgeTerm, move); }
    const double getLogLikelihoodRatioAdjTerm (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioAdjTerm, move); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const double getLogLikelihoodRatioFromBlockMove (const BlockMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogLikelihoodRatioFromBlockMove, move); }
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPriorRatioFromGraphMove, move); }
    const double getLogPriorRatioFromBlockMove (const BlockMove& move) const override { PYBIND11_OVERRIDE(const double, BaseClass, getLogPriorRatioFromBlockMove, move); }
    void applyGraphMove (const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void applyBlockMove (const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyBlockMove, move); }
    void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
    const bool isCompatible(const MultiGraph& graph) const override { PYBIND11_OVERRIDE(bool, BaseClass, isCompatible, graph); }
};

template<typename BaseClass = DegreeCorrectedStochasticBlockModelFamily>
class PyDegreeCorrectedStochasticBlockModelFamily: public PyStochasticBlockModelFamily<BaseClass>{
public:
    using PyStochasticBlockModelFamily<BaseClass>::PyStochasticBlockModelFamily;

    /* Pure abstract methods */

    /* Abstract methods */
    void setBlockPrior(BlockPrior& blockPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setBlockPrior, blockPrior); }
    void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) override { PYBIND11_OVERRIDE(void, BaseClass, setEdgeMatrixPrior, edgeMatrixPrior); }
    void setDegreePrior(DegreePrior& DegreePrior) override { PYBIND11_OVERRIDE(void, BaseClass, setDegreePrior, DegreePrior); }
    const bool isCompatible(const MultiGraph& graph) const override { PYBIND11_OVERRIDE(bool, BaseClass, isCompatible, graph); }
};

}

#endif
