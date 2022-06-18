#ifndef FAST_MIDYNET_HINGE_FLIP_H
#define FAST_MIDYNET_HINGE_FLIP_H


#include "FastMIDyNet/exceptions.h"
#include "util.h"
#include "edge_proposer.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

class HingeFlipProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);
    bool isTrivialMove(const GraphMove&) const;
    const double getLogPropRatioForNormalMove(const GraphMove&) const;
    const double getLogPropRatioForLoopyMove(const GraphMove&) const;
    const double getLogPropRatioForSelfieMove(const GraphMove&) const;
    const double getLogPropRatioForSelfieLoopy(const GraphMove&) const;
protected:
    EdgeSampler m_edgeSampler;
    VertexSampler* m_vertexSamplerPtr = nullptr;
    mutable std::map<BaseGraph::Edge, size_t> m_edgeProposalCounter;
    mutable std::map<BaseGraph::VertexIndex, size_t> m_vertexProposalCounter;
public:
    using EdgeProposer::EdgeProposer;
    const GraphMove proposeRawMove() const override;
    void setUp(const MultiGraph&) override;
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }
    void applyGraphMove(const GraphMove& move) override;
    // void applyBlockMove(const BlockMove& move) override { };
    const double getLogProposalProbRatio(const GraphMove& move) const override ;
    virtual const double getLogVertexWeightRatio(const GraphMove& move) const = 0;

    const std::map<BaseGraph::Edge, size_t>& getEdgeProposalCounts() const {
        return m_edgeProposalCounter;
    }
    const std::map<BaseGraph::VertexIndex, size_t>& getVertexProposalCounts() const {
        return m_vertexProposalCounter;
    }
    void checkSelfSafety() const override {
        EdgeProposer::checkSelfSafety();
        if (m_vertexSamplerPtr == nullptr)
            throw SafetyError("HingeFlipProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
        m_vertexSamplerPtr->checkSafety();
        m_edgeSampler.checkSafety();
    }

    void checkSelfConsistency() const override {
        checkVertexSamplerConsistencyWithGraph("HingeFlipProposer", *m_graphPtr, *m_vertexSamplerPtr);
        checkEdgeSamplerConsistencyWithGraph("HingeFlipProposer", *m_graphPtr, m_edgeSampler);
    }

    void clear() override {
        m_edgeSampler.clear();
        m_vertexSamplerPtr->clear();
    }

};


class HingeFlipUniformProposer: public HingeFlipProposer{
private:
    VertexUniformSampler m_vertexUniformSampler = VertexUniformSampler();
public:
    HingeFlipUniformProposer(bool allowSelfLoops=true, bool allowMultiEdges=true):
        HingeFlipProposer(allowSelfLoops, allowMultiEdges){ m_vertexSamplerPtr = &m_vertexUniformSampler; }
    virtual ~HingeFlipUniformProposer(){}
    const double getLogVertexWeightRatio(const GraphMove& move) const override { return 0; }

};


class HingeFlipDegreeProposer: public HingeFlipProposer{
private:
    VertexDegreeSampler m_vertexDegreeSampler;

public:
    HingeFlipDegreeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double shift=1):
        HingeFlipProposer(allowSelfLoops, allowMultiEdges),
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }
    virtual ~HingeFlipDegreeProposer(){}
    const double getLogVertexWeightRatio(const GraphMove& move) const override {
        BaseGraph::VertexIndex gainingVertex = move.addedEdges[0].second;
        double wk = m_vertexDegreeSampler.getVertexWeight(gainingVertex);
        return log(wk + (move.addedEdges[0].first == move.addedEdges[0].second) ? 2 : 1) - log(wk)
    }
};


} // namespace FastMIDyNet


#endif
