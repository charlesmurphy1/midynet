#ifndef FAST_MIDYNET_LABELED_EDGE_PROPOSER_H
#define FAST_MIDYNET_LABELED_EDGE_PROPOSER_H

#include <algorithm>
#include <iterator>
#include <map>

#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

#include "edge_proposer.h"
#include "util.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"



namespace FastMIDyNet{

using LabelPair = std::pair<BlockIndex,BlockIndex>;

template<class EdgeProposerClass>
class LabeledEdgeProposer: public EdgeProposer {
protected:
    std::map<LabelPair, EdgeProposerClass*> m_proposers;
    mutable LabelPair m_lastSampledLabelPair;
    std::map<LabelPair, size_t> m_weightValues;
    const BlockSequence* m_blocksPtr = nullptr;
    sset::SamplableSet<BaseGraph::Edge> m_edgeSampler = sset::SamplableSet<BaseGraph::Edge>(1, 100);

public:
    using EdgeProposer::EdgeProposer;
    virtual ~LabeledEdgeProposer(){
        clearProposers();
    }

    const LabelPair& getLastSampledLabelPair() const { return m_lastSampledLabelPair; }

    GraphMove proposeRawMove() const override { return proposeRawMove(proposeLabelPair()); }
    GraphMove proposeRawMove(const LabelPair& labelPair) const {
        return m_proposers.at(labelPair)->proposeRawMove();
    }
    const LabelPair proposeLabelPair() const;
    void setUp(const RandomGraph&, std::unordered_set<BaseGraph::VertexIndex> blackList={}) override;
    void setUpFromGraph(const MultiGraph&, std::unordered_set<BaseGraph::VertexIndex> blackList={}) override;
    const double getLogProposalProbRatio(const GraphMove& move) const override {
        return m_proposers.at(m_lastSampledLabelPair)->getLogProposalProbRatio(move);
    };
    void updateProbabilities(const GraphMove& move) override {
        m_proposers[m_lastSampledLabelPair]->updateProbabilities(move);
    };
    void updateProbabilities(const BlockMove& move) override {
        m_proposers[m_lastSampledLabelPair]->updateProbabilities(move);
    };
    virtual EdgeProposerClass* constructNewEdgeProposer() const {
        return new EdgeProposerClass(m_allowSelfLoops, m_allowMultiEdges);
    };
    void clearProposers(){
        for (auto prop : m_proposers){
            delete prop.second;
        }
        m_proposers.clear();
    }

    void getVertexBlackList(
        const std::vector<BlockIndex>& labels, const std::unordered_set<BlockIndex> allowLabels,
        std::unordered_set<BaseGraph::VertexIndex>& blackList
    );

};


/* Definitions */
template<class EdgeProposerSubClass>
const LabelPair LabeledEdgeProposer<EdgeProposerSubClass>::proposeLabelPair() const{
    auto edge = m_edgeSampler.sample_ext_RNG(rng);
    size_t r = (*m_blocksPtr)[edge.first.first], s = (*m_blocksPtr)[edge.first.second];
    m_lastSampledLabelPair = getOrderedEdge({r, s});
    return m_lastSampledLabelPair;
}

template<class EdgeProposerSubClass>
void LabeledEdgeProposer<EdgeProposerSubClass>::setUp(
    const RandomGraph& randomGraph, std::unordered_set<BaseGraph::VertexIndex> blackList
){
    std::map<LabelPair, MultiGraph> subGraphs = getSubGraphOfLabelPair(randomGraph);
    size_t blockCount = randomGraph.getBlockCount();
    m_blocksPtr = &randomGraph.getBlocks();

    clearProposers();
    for (size_t r=0; r<blockCount; ++r){
        for (size_t s=r; s<blockCount; ++s){
            LabelPair labelPair = {r, s};
            std::unordered_set<BaseGraph::VertexIndex> labelBlackList = blackList;
            getVertexBlackList(randomGraph.getBlocks(), {r, s}, labelBlackList);
            m_proposers.insert({labelPair, constructNewEdgeProposer()});
            m_proposers[labelPair]->setUp(randomGraph);
            m_weightValues.insert({labelPair, subGraphs[labelPair].getTotalEdgeNumber()});
        }
    }
    setUpFromGraph(randomGraph.getGraph(), blackList);
}

template<class EdgeProposerSubClass>
void LabeledEdgeProposer<EdgeProposerSubClass>::setUpFromGraph(
    const MultiGraph& graph,
    std::unordered_set<BaseGraph::VertexIndex> blackList
){
    m_graphPtr = &graph;
    m_edgeSampler.clear();
    for (auto vertex: graph) {
        if (blackList.count(vertex) > 0)
            continue;
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (vertex <= neighbor.vertexIndex and blackList.count(neighbor.vertexIndex) == 0)
                m_edgeSampler.insert({vertex, neighbor.vertexIndex}, neighbor.label);
        }
    }
}

template<class EdgeProposerSubClass>
void LabeledEdgeProposer<EdgeProposerSubClass>:: getVertexBlackList(
    const std::vector<BlockIndex>& labels,
    const std::unordered_set<BlockIndex> allowLabels,
    std::unordered_set<BaseGraph::VertexIndex>& blackList
){
    for (size_t i=0; i<labels.size(); ++i){
        if (allowLabels.count(labels[i]) == 0)
            blackList.insert(i);
    }
}

}

#endif
