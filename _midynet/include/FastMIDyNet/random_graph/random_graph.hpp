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
    virtual void _applyGraphMove(const GraphMove&);
public:
    RandomGraph(size_t size=0):
        m_size(size),
        m_graph(size)
        { }
    const MultiGraph& getGraph() const { return m_graph; }

    virtual void setGraph(const MultiGraph& state) {
        m_graph = std::move(state);
    }
    const int getSize() const { return m_size; }
    virtual const size_t& getEdgeCount() const = 0;
    const double getAverageDegree() const {
        double avgDegree = 2 * (double) getEdgeCount();
        avgDegree /= (double) getSize();
        return avgDegree;
    }

    virtual void sample() = 0;
    virtual const double getLogLikelihood() const = 0;
    virtual const double getLogPrior() const = 0;
    const double getLogJoint() const {
        return processRecursiveConstFunction<double>([&](){return getLogLikelihood() + getLogPrior();}, 0);
    }

    virtual const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const = 0;
    virtual const double getLogPriorRatioFromGraphMove (const GraphMove& move) const = 0;
    const double getLogJointRatioFromGraphMove (const GraphMove& move) const{
        return processRecursiveFunction<double>([&](){ return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move); }, 0);
    }
    void applyGraphMove(const GraphMove& move){
        processRecursiveFunction([&](){ _applyGraphMove(move); });
        #if DEBUG
        checkConsistency();
        #endif
    }

    virtual const bool isCompatible(const MultiGraph& graph) const { return graph.getSize() == m_size; }
    virtual bool isSafe() const { return true; }
};

template <typename Label>
class VertexLabeledRandomGraph: public RandomGraph{
protected:
    virtual void _applyLabelMove(const LabelMove<Label>&) { };
public:
    using RandomGraph::RandomGraph;
    virtual const std::vector<Label>& getVertexLabels() const = 0;
    virtual const CounterMap<Label>& getLabelCounts() const = 0;
    virtual const CounterMap<Label>& getEdgeLabelCounts() const = 0;
    virtual const MultiGraph& getLabelGraph() const = 0;
    const Label& getLabelOfIdx(BaseGraph::VertexIndex vertexIdx) const { return getVertexLabels()[vertexIdx]; }

    virtual const double getLogLikelihoodRatioFromLabelMove (const LabelMove<Label>& move) const = 0;
    virtual const double getLogPriorRatioFromLabelMove (const LabelMove<Label>& move) const = 0;
    const double getLogJointRatioFromLabelMove (const LabelMove<Label>& move) const{
        return getLogPriorRatioFromLabelMove(move) + getLogLikelihoodRatioFromLabelMove(move);
    }
    void applyLabelMove(const LabelMove<Label>& move) {
        processRecursiveFunction([&](){ _applyLabelMove(move); });
        #if DEBUG
        checkConsistency();
        #endif
    }
};

} // namespace FastMIDyNet

#endif
