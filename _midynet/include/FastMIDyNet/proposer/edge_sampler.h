#ifndef FASTMIDYNET_EDGE_SAMPLER_H
#define FASTMIDYNET_EDGE_SAMPLER_H

#include <unordered_set>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "BaseGraph/types.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/movetypes.h"

namespace FastMIDyNet{

class EdgeSampler{
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSampler = sset::SamplableSet<BaseGraph::Edge>(1, 100);
    std::vector<double> m_vertexWeights;
public:
    EdgeSampler(){}
    EdgeSampler(const EdgeSampler& other): m_edgeSampler(other.m_edgeSampler){}

    BaseGraph::Edge sample() const {
        return m_edgeSampler.sample_ext_RNG(rng).first;
    }
    void addEdge(const BaseGraph::Edge& );
    void removeEdge(const BaseGraph::Edge& );
    void insertEdge(const BaseGraph::Edge& , double);
    void eraseEdge(const BaseGraph::Edge& );
    void setUp(const MultiGraph&);
    const double getEdgeWeight(const BaseGraph::Edge& edge) const {
        return (m_edgeSampler.count(edge) > 0) ? m_edgeSampler.get_weight(edge) : 0.;
    }
    const double getVertexWeight(const BaseGraph::VertexIndex& vertex) const {
        return m_vertexWeights[vertex];
    }
    const double getTotalWeight() const {return m_edgeSampler.total_weight(); }
    const double getSize() const {return m_edgeSampler.size(); }

    void clear() { m_edgeSampler.clear(); m_vertexWeights.clear(); }
    virtual void checkSafety() const { }
};

}

#endif
