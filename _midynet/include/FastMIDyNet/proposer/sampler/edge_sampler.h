#ifndef FASTMIDYNET_EDGE_SAMPLER_H
#define FASTMIDYNET_EDGE_SAMPLER_H

#include <unordered_set>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "BaseGraph/types.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

class EdgeSampler{
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSampler = sset::SamplableSet<BaseGraph::Edge>(1, 100);
public:
    EdgeSampler(){}
    EdgeSampler(const EdgeSampler& other): m_edgeSampler(other.m_edgeSampler){}

    BaseGraph::Edge sample() const {
        return m_edgeSampler.sample_ext_RNG(rng).first;
    }
    bool contains(const BaseGraph::Edge&edge) const {
        return m_edgeSampler.count(edge) > 0;
    };

    void onEdgeAddition(const BaseGraph::Edge& );
    void onEdgeRemoval(const BaseGraph::Edge& );
    void onEdgeInsertion(const BaseGraph::Edge& , double);
    double onEdgeErasure(const BaseGraph::Edge& );
    const double getEdgeWeight(const BaseGraph::Edge& edge) const {
        // auto orderedEdge = getOrderedEdge(edge);
        return (contains(edge)) ? m_edgeSampler.get_weight(edge) : 0.;
    }

    std::unordered_set<BaseGraph::Edge> enumerateEdges(){
        std::unordered_set<BaseGraph::Edge> edges;
        m_edgeSampler.init_iterator();
        for (size_t i = 0; i < m_edgeSampler.size(); i++) {
            edges.insert(m_edgeSampler.get_at_iterator().first);
            m_edgeSampler.next();
        }
        return edges;
    }

    const double getTotalWeight() const {return m_edgeSampler.total_weight(); }
    const double getSize() const {return m_edgeSampler.size(); }

    void clear() { m_edgeSampler.clear(); }
    void checkSafety() const { }
};

}

#endif
