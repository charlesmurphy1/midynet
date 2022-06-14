#ifndef FAST_MIDYNET_LABELED_DOUBLE_EDGE_SWAP_H
#define FAST_MIDYNET_LABELED_DOUBLE_EDGE_SWAP_H

#include <unordered_map>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "labeled_edge_proposer.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"



namespace FastMIDyNet {

class LabeledDoubleEdgeSwapProposer: public LabeledEdgeProposer {
private:
    mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    std::unordered_map<LabelPair, EdgeSampler*> m_labeledEdgeSampler;
public:
    LabeledDoubleEdgeSwapProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double labelPairShift=1):
        LabeledEdgeProposer(allowSelfLoops, allowMultiEdges, labelPairShift) { }
    virtual ~LabeledDoubleEdgeSwapProposer(){ clear(); }
    const GraphMove proposeRawMove() const override ;
    void setUpFromGraph(const MultiGraph& graph) override ;
    const double getLogProposalProbRatio(const GraphMove& move) const override { return 0; }
    void applyGraphMove(const GraphMove& move) override ;
    void applyBlockMove(const BlockMove& move) override ;
    size_t getTotalEdgeCount() const ;
    void clear(){
        LabeledEdgeProposer::clear();
        for (auto p : m_labeledEdgeSampler)
            delete p.second;
        m_labeledEdgeSampler.clear();
    }

    void checkSelfConsistency() const override {
        for (auto vertex : *m_graphPtr){
            for (auto neighbor : m_graphPtr->getNeighboursOfIdx(vertex)){
                if (vertex > neighbor.vertexIndex)
                    continue;
                const auto rs = getOrderedEdge({
                    m_labelSampler.getLabelOfIdx(vertex), m_labelSampler.getLabelOfIdx(neighbor.vertexIndex)
                });
                const auto edge = getOrderedEdge({ vertex, neighbor.vertexIndex });

                if (not m_labeledEdgeSampler.at(rs)->contains(edge))
                    throw ConsistencyError(
                        "LabeledDoubleEdgeSwapProposer: graph is inconsistent with edge sampler, edge ("
                        + std::to_string(edge.first) + ", "
                        + std::to_string(edge.second) + ") is missing."
                    );
            }
        }
    }
};


} // namespace FastMIDyNet


#endif
