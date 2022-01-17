#ifndef FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP
#define FAST_MIDYNET_PYTHON_RANDOMGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
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
    const std::vector<BlockIndex>& getBlocks() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<BlockIndex>&, BaseClass, getBlocks, );
    }
    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex vertexIdx) const override  {
        PYBIND11_OVERRIDE_PURE(const BlockIndex&, BaseClass, getBlockOfIdx, vertexIdx);
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
    const std::vector<size_t>& getDegreeOfIdx(BaseGraph::VertexIndex idx) const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, BaseClass, getDegreeOfIdx, idx);
    }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const override  {
        PYBIND11_OVERRIDE_PURE(const std::vector<CounterMap<size_t>>&, BaseClass, getDegreeCountsInBlocks, );
    }

    /* Abstract methods */
    const BlockSequence& getLabels() const override { PYBIND11_OVERRIDE(const BlockSequence&, BaseClass, getLabels, ); }
    const BlockIndex& getLabelOfIdx(BaseGraph::VertexIndex vertexIdx) const override {
        PYBIND11_OVERRIDE(const BlockIndex&, BaseClass, getLabelOfIdx, vertexIdx);
    }
    void setState(const MultiGraph& graph) override { PYBIND11_OVERRIDE(void, BaseClass, setState, graph); }
    void applyMove(const GraphMove& move) override { PYBIND11_OVERRIDE(void, BaseClass, applyMove, move); }
    void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
};

}

#endif
