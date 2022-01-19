#ifndef FAST_MIDYNET_SINGLE_EDGE_H
#define FAST_MIDYNET_SINGLE_EDGE_H


#include "FastMIDyNet/exceptions.h"
#include "edge_proposer.h"
#include "vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class SingleEdgeProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_addOrRemoveDistribution = std::bernoulli_distribution(.5);
protected:
    VertexSampler* m_vertexSamplerPtr = NULL;
    const FastMIDyNet::MultiGraph* m_graphPtr = NULL;
public:
    using EdgeProposer::EdgeProposer;
    GraphMove proposeRawMove() const override;
    void setUp(const RandomGraph& randomGraph) override { EdgeProposer::setUp(randomGraph); setUp(randomGraph.getGraph()); }
    void setUp(const MultiGraph&);
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }
    void updateProbabilities(const GraphMove& move) override { };
    void updateProbabilities(const BlockMove& move) override { };
    void checkSafety() const override {
        if (m_graphPtr == nullptr)
            throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_graphPtr` is NULL.");
        if (m_vertexSamplerPtr == nullptr)
            throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
    }
};

class SingleEdgeUniformProposer: public SingleEdgeProposer{
private:
    VertexUniformSampler m_vertexUniformSampler;
public:
    SingleEdgeUniformProposer(bool allowSelfLoops=true, bool allowMultiEdges=true):
        SingleEdgeProposer(allowSelfLoops, allowMultiEdges){ m_vertexSamplerPtr = &m_vertexUniformSampler; }

    const double getLogProposalProbRatio(const GraphMove&move) const override{
        double logProbability = 0;

        for (auto edge: move.removedEdges)
            if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
                logProbability += -log(.5);

        for (auto edge: move.addedEdges)
            if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
                logProbability += -log(.5);
        return logProbability;
    }
};

class SingleEdgeDegreeProposer: public SingleEdgeProposer{
private:
    VertexDegreeSampler m_vertexDegreeSampler;
public:
    SingleEdgeDegreeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double shift=1):
        SingleEdgeProposer(allowSelfLoops, allowMultiEdges),
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }

    const double getLogProposalProbRatio(const GraphMove&move) const override{
        double logProbability = 0;

        for (auto edge: move.removedEdges){
            if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 1)
                logProbability += -log(.5);

            logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.first) - 1);
            logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.second) - 1);
            logProbability -= log(m_vertexDegreeSampler.getTotalWeight() - 1);

            logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.first));
            logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.second));
            logProbability += log(m_vertexDegreeSampler.getTotalWeight());
        }
        for (auto edge: move.addedEdges){
            if (m_graphPtr->getEdgeMultiplicityIdx(edge) == 0)
                logProbability += -log(.5);

            logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.first) + 1);
            logProbability += log(m_vertexDegreeSampler.getVertexWeight(edge.second) + 1);
            logProbability -= log(m_vertexDegreeSampler.getTotalWeight() + 1);

            logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.first));
            logProbability -= log(m_vertexDegreeSampler.getVertexWeight(edge.second));
            logProbability += log(m_vertexDegreeSampler.getTotalWeight());
        }
        return logProbability;
    }
};


} // namespace FastMIDyNet


#endif
