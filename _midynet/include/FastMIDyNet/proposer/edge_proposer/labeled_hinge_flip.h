#ifndef FAST_MIDYNET_LABELED_HINGE_FLIP_H
#define FAST_MIDYNET_LABELED_HINGE_FLIP_H

#include <map>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "labeled_edge_proposer.h"
#include "FastMIDyNet/proposer/vertex_sampler.h"
#include "FastMIDyNet/proposer/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"



namespace FastMIDyNet {

class LabeledHingeFlipProposer: public LabeledEdgeProposer {
private:
    mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    std::map<LabelPair, EdgeSampler*> m_labeledEdgeSampler;
    std::map<BlockIndex, VertexSampler*> m_labeledVertexSampler;
public:
    // using LabeledEdgeProposer::LabeledEdgeProposer;
    LabeledHingeFlipProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double labelPairShift=1):
        LabeledEdgeProposer(allowSelfLoops, allowMultiEdges, labelPairShift) { }
    ~LabeledHingeFlipProposer(){ clear(); }
    GraphMove proposeRawMove() const override ;
    void setUpFromGraph(const MultiGraph& graph) override ;
    const double getLogProposalProbRatio(const GraphMove& move) const override ;
    void applyGraphMove(const GraphMove& move) override ;
    void applyBlockMove(const BlockMove& move) override ;
    virtual VertexSampler* constructVertexSampler() const = 0;
    size_t getTotalEdgeCount() const {
        size_t edgeCount = 0;
        for (auto s: m_labeledEdgeSampler)
            edgeCount += s.second->getTotalWeight();
        return edgeCount;
    }
    void clear(){
        for (auto p : m_labeledEdgeSampler)
            delete p.second;
        m_labeledEdgeSampler.clear();

        for (auto p : m_labeledVertexSampler)
            delete p.second;
        m_labeledVertexSampler.clear();
    };
};


} // namespace FastMIDyNet


#endif
