#ifndef FAST_MIDYNET_SINGLE_EDGE_H
#define FAST_MIDYNET_SINGLE_EDGE_H


#include "FastMIDyNet/exceptions.h"
#include "edge_proposer.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class SingleEdgeProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_addOrRemoveDistribution = std::bernoulli_distribution(.5);
protected:
    VertexSampler* m_vertexSamplerPtr = nullptr;
public:
    using EdgeProposer::EdgeProposer;
    GraphMove proposeRawMove() const override;
    void setUpFromGraph(const MultiGraph&) override;
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }
    virtual void applyGraphMove(const GraphMove& move) override { };
    void applyBlockMove(const BlockMove& move) override { };

    void checkSelfSafety() const override {
        EdgeProposer::checkSelfSafety();
        if (m_graphPtr == nullptr)
            throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_graphPtr` is NULL.");
        if (m_vertexSamplerPtr == nullptr)
            throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
    }
    void clear() { m_vertexSamplerPtr->clear(); }
};

class SingleEdgeUniformProposer: public SingleEdgeProposer{
private:
    VertexUniformSampler m_vertexUniformSampler;
public:
    SingleEdgeUniformProposer(bool allowSelfLoops=true, bool allowMultiEdges=true):
        SingleEdgeProposer(allowSelfLoops, allowMultiEdges){ m_vertexSamplerPtr = &m_vertexUniformSampler; }
    virtual ~SingleEdgeUniformProposer(){}

    const double getLogProposalProbRatio(const GraphMove&move) const override;
};

class SingleEdgeDegreeProposer: public SingleEdgeProposer{
private:
    VertexDegreeSampler m_vertexDegreeSampler;
public:
    SingleEdgeDegreeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double shift=1):
        SingleEdgeProposer(allowSelfLoops, allowMultiEdges),
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }

    virtual ~SingleEdgeDegreeProposer(){}

    void applyGraphMove(const GraphMove& move) override ;
    const double getLogProposalProbRatio(const GraphMove&move) const override;

    const double getGammaRatio(BaseGraph::Edge edge, const double difference=1) const ;
};


} // namespace FastMIDyNet


#endif
