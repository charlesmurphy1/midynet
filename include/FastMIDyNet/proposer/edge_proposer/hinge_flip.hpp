#ifndef FAST_MIDYNET_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_DOUBLE_EDGE_SWAP_H


#include "edge_proposer.h"
#include "vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

/* Prototypes */
template<typename VertexSamplerType = VertexUniformSampler>
class HingeFlip: public EdgeProposer {
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    // sset::SamplableSet<BaseGraph::VertexIndex> m_nodeSamplableSet = sset::SamplableSet<BaseGraph::VertexIndex> (1, 100);
    const VertexSamplerType* m_vertexSamplerPtr = NULL;
    std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);

public:
    using EdgeProposer::EdgeProposer;
    HingeFlip(){
        m_vertexSamplerPtr = new VertexSamplerType();
    }
    ~HingeFlip(){
        delete m_vertexSamplerPtr;
    }
    void acceptIsolated(bool accept);

    GraphMove proposeMove();
    void setUp(const RandomGraph& randomGraph) { setUp(randomGraph.getState()); }
    void setUp(const MultiGraph& graph);
;
    double getLogProposalProbRatio(const GraphMove&) const { return 0; }
    void updateProbabilities(const GraphMove& move);

    // For tests
    const sset::SamplableSet<BaseGraph::Edge>& getEdgeSamplableSet() { return m_edgeSamplableSet; }
    const sset::SamplableSet<BaseGraph::VertexIndex>& getNodeSamplableSet() { return m_nodeSamplableSet; }
};

/* Definitions */
template<typename VertexSamplerType = VertexUniformSampler>
GraphMove HingeFlip<VertexSamplerType>::proposeMove() {
    auto edge = m_edgeSamplableSet.sample_ext_RNG(rng).first;
    BaseGraph::VertexIndex node = m_vertexSamplerPtr->sample(rng);

    if (edge.first == node or edge.second == node)
        return GraphMove();

    BaseGraph::Edge newEdge;
    if (m_flipOrientationDistribution(rng)) {
        newEdge = {edge.first, node};
    }
    else {
        newEdge = {edge.second, node};
    }
    return {{edge}, {newEdge}};
};

template<typename VertexSamplerType = VertexUniformSampler>
bool HingeFlip<VertexSamplerType>::acceptIsolated() {
    m_withIsolatedVertices = accept; m_vertexSamplerPtr->acceptIsolated(accept);
    return m_withIsolatedVertices;
}

template<typename VertexSamplerType = VertexUniformSampler>
void HingeFlip<VertexSamplerType>::setUp(const MultiGraph& graph){
    m_vertexSamplerPtr->setUp(graph);
    for (auto vertex: graph) {
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex <= neighbor.vertexIndex)
                m_edgeSamplableSet.insert({vertex, neighbor.vertexIndex}, neighbor.label);
        }
    }
}

template<typename VertexSamplerType = VertexUniformSampler>
void HingeFlip<VertexSamplerType>::updateProbabilities(const GraphMove& move) {
    m_vertexSamplerPtr->update(move);

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


} // namespace FastMIDyNet


#endif
