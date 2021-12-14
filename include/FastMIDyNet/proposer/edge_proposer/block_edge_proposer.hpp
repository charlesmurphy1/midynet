#ifndef FAST_MIDYNET_BLOCK_EDGE_PROPOSER_H
#define FAST_MIDYNET_BLOCK_EDGE_PROPOSER_H

#include <map>

#include "SamplableSet.hpp"
#include "FastMIDyNet/proposer/edge_proposer/edge_proposer.h"
#include "FastMIDyNet/proposer/edge_proposer/single_edge_move.h"
#include "FastMIDyNet/random_graph/sbm.h"

namespace FastMIDyNet{

template<typename EdgeProposerType>
class BlockPreservingEdgeProposer: public EdgeProposer{
protected:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    std::map<std::pair<BlockIndex, BlockIndex>, EdgeProposerType> m_edgeProposerByBlockPair;
    const BlockSequence* m_blocksPtr = NULL;
    size_t m_blockCount;
public:
    BlockPreservingEdgeProposer(const BlockSequence& blocks):
        m_blocksPtr(&blocks),
        m_blockCount(*max_element(blocks.begin(), blocks.end()) + 1){}
    BlockPreservingEdgeProposer(){}
    GraphMove proposeMove(){
        auto edge = m_edgeSamplableSet.sample_ext_RNG(rng).first;
        BlockIndex r = (*m_blocksPtr)[edge.first], s = (*m_blocksPtr)[edge.second];
        return m_edgeProposerByBlockPair[getOrderedPair<BlockIndex>({r, s})].proposeMove();
    }

    void setBlocks(const BlockSequence& blocks){
        m_blocksPtr = &blocks;
        m_blockCount = *max_element(blocks.begin(), blocks.end()) + 1;
    }

    void setUp(const RandomGraph& randomGraph) {
        setUp(randomGraph.getState());
    }

    void setUp(const MultiGraph& graph){
        for (auto vertex: graph) {
            for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
                if (vertex <= neighbor.vertexIndex)
                    m_edgeSamplableSet.insert({vertex, neighbor.vertexIndex}, neighbor.label);
            }
        }

        for(size_t r=0; r < m_blockCount; ++r){ for(size_t s=r+1; s < m_blockCount; ++s){
            m_edgeProposerByBlockPair.insert({{r, s}, EdgeProposerType()});
            m_edgeProposerByBlockPair[{r, s}].acceptIsolated(false);
            m_edgeProposerByBlockPair[{r, s}].setUp(getSubGraphByBlocks(graph, *m_blocksPtr, r, s));
        }}
    }

    double getLogProposalProbRatio(const GraphMove& move) const{
        BlockIndex r, s;

        if ( move.addedEdges.size() > 0 ){
            r = (*m_blocksPtr)[move.addedEdges.begin()->first];
            s = (*m_blocksPtr)[move.addedEdges.begin()->second];
        } else if ( move.removedEdges.size() > 0 ){
            r = (*m_blocksPtr)[move.removedEdges.begin()->first];
            s = (*m_blocksPtr)[move.removedEdges.begin()->second];
        } else {
            return 0;
        }
        auto index = getOrderedPair<BlockIndex>({r, s});
        return m_edgeProposerByBlockPair.at(index).getLogProposalProbRatio(move);
    }


    void updateProbabilities(const GraphMove& move) {
        size_t edgeWeight;
        BaseGraph::Edge edge;
        for (auto removedEdge: move.removedEdges) {
            edge = getOrderedEdge(removedEdge);
            edgeWeight = round(m_edgeSamplableSet.get_weight(edge));
            if (edgeWeight == 1)
                m_edgeSamplableSet.erase(edge);
            else
                m_edgeSamplableSet.set_weight(edge, edgeWeight-1);
        }

        for (auto addedEdge: move.addedEdges) {
            edge = getOrderedEdge(addedEdge);
            if (m_edgeSamplableSet.count(edge) == 0)
                m_edgeSamplableSet.insert(edge, 1);
            else {
                edgeWeight = round(m_edgeSamplableSet.get_weight(edge));
                m_edgeSamplableSet.set_weight(edge, edgeWeight+1);
            }
        }
    }
};



}

#endif /* FastMIDyNet */
