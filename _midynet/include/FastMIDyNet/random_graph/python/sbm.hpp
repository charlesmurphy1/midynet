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
    double getLogLikelihood() const override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihood, ); }
    double getLogPrior() const override { PYBIND11_OVERRIDE(double, BaseClass, getLogPrior, ); }
    double getLogLikelihoodRatioEdgeTerm (const GraphMove& move) const override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatioEdgeTerm, move); }
    double getLogLikelihoodRatioAdjTerm (const GraphMove& move) const override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatioAdjTerm, move); }
    double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    double getLogLikelihoodRatioFromBlockMove (const BlockMove& move) const override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatioFromBlockMove, move); }
    double getLogPriorRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE(double, BaseClass, getLogPriorRatioFromGraphMove, move); }
    double getLogPriorRatioFromBlockMove (const BlockMove& move) const override { PYBIND11_OVERRIDE(double, BaseClass, getLogPriorRatioFromBlockMove, move); }
    void applyGraphMove (const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void applyBlockMove (const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyBlockMove, move); }
    void computationFinished() const override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
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
};

}

#endif
