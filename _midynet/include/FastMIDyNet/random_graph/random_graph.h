#ifndef FAST_MIDYNET_RANDOM_GRAPH_H
#define FAST_MIDYNET_RANDOM_GRAPH_H

// #include <random>
#include <vector>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/utility/maps.hpp"


namespace FastMIDyNet{

class RandomGraph: public NestedRandomVariable{
protected:
    size_t m_size;
    MultiGraph m_graph;
    virtual void samplePriors() = 0;

    virtual void _applyGraphMove(const GraphMove&);
    virtual void _applyBlockMove(const BlockMove&) { };
public:

    RandomGraph(size_t size=0):
        m_size(size),
        m_graph(size)
        { }

    const MultiGraph& getGraph() const { return m_graph; }
    virtual void setGraph(const MultiGraph& state) {
        m_graph = state;
    }

    const int getSize() const { return m_size; }
    const double getAverageDegree() const {
        double avgDegree = 2 * (double) getEdgeCount();
        avgDegree /= (double) getSize();
        return avgDegree;
    }
    virtual const std::vector<BlockIndex>& getBlocks() const = 0;
    virtual const size_t& getBlockCount() const = 0;
    virtual const std::vector<size_t>& getVertexCountsInBlocks() const = 0;
    virtual const Matrix<size_t>& getEdgeMatrix() const = 0;
    virtual const std::vector<size_t>& getEdgeCountsInBlocks() const = 0;
    virtual const size_t& getEdgeCount() const = 0;
    virtual const std::vector<size_t>& getDegrees() const = 0;
    virtual const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const = 0;
    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex vertexIdx) const { return getBlocks()[vertexIdx]; }
    const size_t& getDegreeOfIdx(BaseGraph::VertexIndex vertexIdx) const { return getDegrees()[vertexIdx]; }
    virtual const bool isCompatible(const MultiGraph& graph) const { return graph.getSize() == m_size; }

    const size_t computeBlockCount() const ;
    const std::vector<size_t> computeVertexCountsInBlocks() const ;
    const Matrix<size_t> computeEdgeMatrix() const ;
    const std::vector<size_t> computeEdgeCountsInBlocks() const ;
    const std::vector<CounterMap<size_t>> computeDegreeCountsInBlocks() const ;


    const MultiGraph& sample() {
        samplePriors();
        sampleGraph();
        #if DEBUG
        checkSelfConsistency();
        #endif
        return getGraph();
    }
    virtual void sampleGraph() = 0;
    virtual const double getLogLikelihood() const = 0;
    virtual const double getLogPrior() const = 0;
    const double getLogJoint() const { return getLogLikelihood() + getLogPrior(); }

    virtual const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const = 0;
    virtual const double getLogLikelihoodRatioFromBlockMove (const BlockMove& move) const = 0;
    virtual const double getLogPriorRatioFromGraphMove (const GraphMove& move) const = 0;
    virtual const double getLogPriorRatioFromBlockMove (const BlockMove& move) const = 0;
    const double getLogJointRatioFromGraphMove (const GraphMove& move) const{
        return processRecursiveFunction<double>([&](){ return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move); }, 0);
    }
    const double getLogJointRatioFromBlockMove (const BlockMove& move) const{
        return getLogPriorRatioFromBlockMove(move) + getLogLikelihoodRatioFromBlockMove(move);
    }
    void applyGraphMove(const GraphMove& move){
        processRecursiveFunction([&](){ _applyGraphMove(move); });
        #if DEBUG
        checkConsistency();
        #endif
    }
    void applyBlockMove(const BlockMove& move) {
        processRecursiveFunction([&](){ _applyBlockMove(move); });
        #if DEBUG
        checkConsistency();
        #endif
    }
};

} // namespace FastMIDyNet

#endif
