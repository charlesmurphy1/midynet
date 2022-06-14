#ifndef FAST_MIDYNET_LABELED_HINGE_FLIP_H
#define FAST_MIDYNET_LABELED_HINGE_FLIP_H

#include <unordered_map>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "labeled_edge_proposer.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "FastMIDyNet/utility/functions.h"



namespace FastMIDyNet {

class LabeledHingeFlipProposer: public LabeledEdgeProposer {
private:
    mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    std::unordered_map<LabelPair, EdgeSampler*> m_labeledEdgeSampler;
    std::unordered_map<BlockIndex, VertexSampler*> m_labeledVertexSampler;
public:
    LabeledHingeFlipProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double labelPairShift=1):
        LabeledEdgeProposer(allowSelfLoops, allowMultiEdges, labelPairShift) { }
    virtual ~LabeledHingeFlipProposer(){ clear(); }
    const GraphMove proposeRawMove() const override ;
    void setUpFromGraph(const MultiGraph& graph) override ;
    void applyGraphMove(const GraphMove& move) override ;
    void applyBlockMove(const BlockMove& move) override ;
    virtual VertexSampler* constructVertexSampler() const = 0;
    size_t getTotalEdgeCount() const {
        size_t edgeCount = 0;
        for (auto s: m_labeledEdgeSampler)
            edgeCount += s.second->getTotalWeight();
        return edgeCount;
    }
    void clear() override;
};

class LabeledHingeFlipUniformProposer: public LabeledHingeFlipProposer {
public:
    LabeledHingeFlipUniformProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double labelPairShift=1):
        LabeledHingeFlipProposer(allowSelfLoops, allowMultiEdges, labelPairShift) { }
    VertexSampler* constructVertexSampler() const override {
        return new VertexUniformSampler();
    }
    const double getLogProposalProbRatio(const GraphMove& move) const override { return 0; }
};

class LabeledHingeFlipDegreeProposer: public LabeledHingeFlipProposer {
protected:
    double m_vertexShift;
public:
    LabeledHingeFlipDegreeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double labelPairShift=1, double vertexShift=1):
        LabeledHingeFlipProposer(allowSelfLoops, allowMultiEdges, labelPairShift), m_vertexShift(vertexShift) { }
    VertexSampler* constructVertexSampler() const override {
        return new VertexDegreeSampler(m_vertexShift);
    }
    const double getLogProposalProbRatio(const GraphMove& move) const override ;
};


} // namespace FastMIDyNet


#endif
