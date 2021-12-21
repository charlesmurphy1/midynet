#ifndef FAST_MIDYNET_PYTHON_SBM_HPP
#define FAST_MIDYNET_PYTHON_SBM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/random_graph/python/randomgraph.hpp"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{
//
template<typename BaseClass = StochasticBlockModelFamily>
class PyStochasticBlockModelFamily: public PyRandomGraph<StochasticBlockModelFamily>{
public:
    using PyRandomGraph<StochasticBlockModelFamily>::PyRandomGraph;
    
    /* Pure abstract methods */

    /* Abstract methods */
    double getLogLikelihood() const override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihood, ); }
    double getLogPrior()  override { PYBIND11_OVERRIDE(double, BaseClass, getLogPrior, ); }
    double getLogLikelihoodRatioEdgeTerm (const GraphMove& move)  override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatioEdgeTerm, move); }
    double getLogLikelihoodRatioAdjTerm (const GraphMove& move)  override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatioAdjTerm, move); }
    double getLogLikelihoodRatio (const GraphMove& move)  override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatio, move); }
    double getLogLikelihoodRatio (const BlockMove& move)  override { PYBIND11_OVERRIDE(double, BaseClass, getLogLikelihoodRatio, move); }
    double getLogPriorRatio (const GraphMove& move)  override { PYBIND11_OVERRIDE(double, BaseClass, getLogPriorRatio, move); }
    double getLogPriorRatio (const BlockMove& move)  override { PYBIND11_OVERRIDE(double, BaseClass, getLogPriorRatio, move); }
    void applyMove (const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyMove, move); }
    void applyMove (const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyMove, move); }
    void computationFinished()  override { PYBIND11_OVERRIDE(void, BaseClass, computationFinished, ); }
    void checkSelfConsistency()  override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
};

}

#endif
