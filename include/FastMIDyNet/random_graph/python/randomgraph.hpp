#ifndef FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP
#define FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = RandomGraph>
class PyRandomGraph: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */
    void sampleState() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleState, ); }
    void samplePriors() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, samplePriors, ); }
    double getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogLikelihood, ); }
    double getLogPrior() override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogPrior, ); }
    double getLogLikelihoodRatio (const GraphMove& move) override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogLikelihoodRatio, move); }
    double getLogPriorRatio (const GraphMove& move) override { PYBIND11_OVERRIDE_PURE(double, BaseClass, getLogPriorRatio, move); }

    /* Abstract methods */
    void setState(const MultiGraph& graph) override { PYBIND11_OVERRIDE(void, BaseClass, setState, graph); }
    void applyMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyMove, move); }
    void checkSelfConsistency() override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
};

}

#endif
