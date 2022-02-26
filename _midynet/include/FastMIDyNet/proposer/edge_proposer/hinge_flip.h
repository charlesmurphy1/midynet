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
    const double getLogProposalProbRatio(const GraphMove& move) const override {
        if (getOrderedEdge(move.addedEdges[0]) == getOrderedEdge(move.removedEdges[0]))
            return 0.;
        double weight = getLogProposalWeight(move);
        GraphMove reversedMove = getReverseMove(move);
        double reversedWeight = getLogReverseProposalWeight(move);
        return reversedWeight - weight;
    }
    const double getLogProposalWeight(const GraphMove& move) const {
        auto gainingVertex = move.addedEdges[0].second;
        double edgeWeight = m_edgeSampler.getEdgeWeight(getOrderedEdge(move.removedEdges[0]));
        double vertexWeight = m_vertexSamplerPtr->getVertexWeight(gainingVertex);
        return log(edgeWeight) + log(vertexWeight);
    }
    const double getLogReverseProposalWeight(const GraphMove& move) const {
        auto losingVertex = move.removedEdges[0].second;
        double edgeWeight = m_edgeSampler.getEdgeWeight(getOrderedEdge(move.addedEdges[0])) + 1;
        double vertexWeight = m_vertexSamplerPtr->getVertexWeight(losingVertex);
        return log(edgeWeight) + log(vertexWeight);
    }

    const std::map<BaseGraph::Edge, size_t>& getEdgeProposalCounts() const {
        return m_edgeProposalCounter;
    }
    const std::map<BaseGraph::VertexIndex, size_t>& getVertexProposalCounts() const {
        return m_vertexProposalCounter;
    }
    void checkSafety() const override {
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
    void checkConsistency() const override {
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

    void checkConsistency() const override { }
};


} // namespace FastMIDyNet


#endif
