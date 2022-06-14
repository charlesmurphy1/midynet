#ifndef FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP
#define FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/python/rv.hpp"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/proposer/movetypes.h"


namespace FastMIDyNet{

template<typename BaseClass = RandomGraph>
class PyRandomGraph: public PyNestedRandomVariable<BaseClass>{
protected:
    void samplePriors() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, samplePriors, ); }
    void _applyGraphMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMove, move); }
    void _applyBlockMove(const BlockMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyBlockMove, move); }
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
    
    /* Pure abstract methods */
    void sampleGraph() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleGraph, ); }
    const double getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, ); }
    const double getLogPrior() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPrior, ); }
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    const double getLogLikelihoodRatioFromBlockMove (const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromBlockMove, move); }
    const double getLogPriorRatioFromGraphMove (const GraphMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPriorRatioFromGraphMove, move); }
    const double getLogPriorRatioFromBlockMove (const BlockMove& move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPriorRatioFromBlockMove, move); }

    const std::vector<BlockIndex>& getBlocks() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<BlockIndex>&, BaseClass, getBlocks, );
    }
    const size_t& getBlockCount() const override  {
        PYBIND11_OVERRIDE_PURE(const size_t&, BaseClass, getBlockCount, );
    }
    const std::vector<size_t>& getVertexCountsInBlocks() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, BaseClass, getVertexCountsInBlocks, );
    }
    const Matrix<size_t>& getEdgeMatrix() const override  {
        PYBIND11_OVERRIDE_PURE(const Matrix<size_t>&, BaseClass, getEdgeMatrix, );
    }
    const std::vector<size_t>& getEdgeCountsInBlocks() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, BaseClass, getEdgeCountsInBlocks, );
    }
    const size_t& getEdgeCount() const override  {
        PYBIND11_OVERRIDE_PURE(const size_t&, BaseClass, getEdgeCount, );
    }
    const std::vector<size_t>& getDegrees() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, BaseClass, getDegrees, );
    }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<CounterMap<size_t>>&, BaseClass, getDegreeCountsInBlocks, );
    }

    /* Abstract methods */
    void setGraph(const MultiGraph& graph) override { PYBIND11_OVERRIDE(void, BaseClass, setGraph, graph); }
    const bool isCompatible(const MultiGraph& graph) const override { PYBIND11_OVERRIDE(bool, BaseClass, isCompatible, graph); }
    bool isSafe() const override { PYBIND11_OVERRIDE(bool, BaseClass, isSafe, ); }
};

}

#endif
