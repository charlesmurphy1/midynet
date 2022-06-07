#ifndef FAST_MIDYNET_HINGE_FLIP_H
#define FAST_MIDYNET_HINGE_FLIP_H


#include "FastMIDyNet/exceptions.h"
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
    GraphMove proposeRawMove() const override;
    void setUpFromGraph(const MultiGraph&) override;
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }
    void applyGraphMove(const GraphMove& move) override;
    void applyBlockMove(const BlockMove& move) override { };
    const double getLogProposalProbRatio(const GraphMove& move) const override ;
    virtual const double getLogVertexWeightRatio(const GraphMove& move) const = 0;

    const std::map<BaseGraph::Edge, size_t>& getEdgeProposalCounts() const {
        return m_edgeProposalCounter;
    }
    const std::map<BaseGraph::VertexIndex, size_t>& getVertexProposalCounts() const {
        return m_vertexProposalCounter;
    }
    void checkSelfSafety() const override {
        if (m_vertexSamplerPtr == nullptr)
            throw SafetyError("HingeFlipProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
        m_vertexSamplerPtr->checkSafety();
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

    void checkSelfConsistency() const override {
        for (auto vertex : *m_graphPtr)
            if (not m_vertexUniformSampler.contains(vertex))
                throw ConsistencyError(
                    "HingeFlipUniformProposer: vertexSampler is inconsistent with graph, "
                    + std::to_string(vertex) + " is not in sampler."
                );
    }

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
        if (move.addedEdges[0].first == move.addedEdges[0].second)
            return log(wk + 2) - log(wk);
        else
            return log(wk + 1) - log(wk);
    }

    void checkSelfConsistency() const override { }
};


} // namespace FastMIDyNet


#endif
