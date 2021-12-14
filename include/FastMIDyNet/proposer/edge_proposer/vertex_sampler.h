#ifndef FAST_MIDYNET_VERTEX_SAMPLER_H
#define FAST_MIDYNET_VERTEX_SAMPLER_H

#include <random>

#include "BaseGraph/types.h"
#include "SamplableSet.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class VertexSampler{
protected:
    bool m_withIsolatedVertices = true;
public:
    virtual BaseGraph::VertexIndex const sample() = 0;
    virtual void setUp(const MultiGraph&) = 0;
    virtual void update(const GraphMove&) = 0;
    virtual double getLogProposalProbRatio(const GraphMove&) = 0;

    const bool& acceptIsolated() const { return m_withIsolatedVertices; }
    void acceptIsolated(bool accept) { m_withIsolatedVertices = accept; }
};

class VertexUniformSampler: public VertexSampler{
private:
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler = sset::SamplableSet<BaseGraph::VertexIndex>(1, 100);
public:
    BaseGraph::VertexIndex const sample(){
        return m_vertexSampler.sample_ext_RNG(rng).first;
    }
    void setUp(const MultiGraph& graph){
        for (auto vertex : graph){
            if (m_withIsolatedVertices or graph.getDegreeOfIdx(vertex) > 0)
                m_vertexSampler.insert(vertex, 1);
        }
    }
    void update(const GraphMove& move) {}
};

class VertexDegreeSampler: public VertexSampler{
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSampler = sset::SamplableSet<BaseGraph::Edge>(1, 100);
    std::bernoulli_distribution m_vertexChoiceDistribution = std::bernoulli_distribution(.5);
    double m_shift;
public:

    VertexDegreeSampler(double shift=1):m_shift(shift){};
    BaseGraph::VertexIndex const sample(){
        auto edge = m_edgeSampler.sample_ext_RNG(rng).first;

        if ( m_vertexChoiceDistribution(rng) )
            return edge.first;
        else
            return edge.second;
    }
    void setUp(const MultiGraph& graph){
        m_edgeSamplableSet.clear();
        for (auto vertex: graph)
            for (auto neighbor: graph.getNeighboursOfIdx(vertex))
                if (vertex <= neighbor.vertexIndex)
                    m_edgeSamplableSet.insert({vertex, neighbor.vertexIndex}, neighbor.label);
    }

    void update(const GraphMove& move) {
        for (auto edge: move.removedEdges) {
            edge = getOrderedEdge(edge);
            size_t edgeWeight = round(m_edgeSamplableSet.get_weight(edge));
            if (edgeWeight == 1)
                m_edgeSamplableSet.erase(edge);
            else
                m_edgeSamplableSet.set_weight(edge, edgeWeight-1);
        }

        for (auto edge: move.addedEdges) {
            edge = getOrderedEdge(edge);
            if (m_edgeSamplableSet.count(edge) == 0)
                m_edgeSamplableSet.insert(edge, 1);
            else {
                m_edgeSamplableSet.set_weight(edge, round(m_edgeSamplableSet.get_weight(edge))+1);
            }
        }
    }

};

}/* FastMIDyNet */

#endif
