#ifndef FAST_MIDYNET_SINGLE_EDGE_H
#define FAST_MIDYNET_SINGLE_EDGE_H


#include "edge_proposer.h"
#include "vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"


namespace FastMIDyNet {

class SingleEdgeProposer: public EdgeProposer {
private:
    const FastMIDyNet::MultiGraph* m_graphPtr = NULL;
    std::bernoulli_distribution m_addOrRemoveDistribution = std::bernoulli_distribution(.5);
protected:
    VertexSampler* m_vertexSamplerPtr = NULL;
public:
    GraphMove proposeMove();
    void setUp(const RandomGraph& randomGraph) { setUp(randomGraph.getState()); }
    void setUp(const MultiGraph&);
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }
    double getLogProposalProbRatio(const GraphMove&) const;
    void updateProbabilities(const GraphMove&) { }
};

class SingleEdgeUniformProposer: SingleEdgeProposer{
private:
    VertexUniformSampler m_vertexUniformSampler;
public:
    SingleEdgeUniformProposer(){ m_vertexSamplerPtr = &m_vertexUniformSampler; }
};

class SingleEdgeDegreeProposer: SingleEdgeProposer{
private:
    VertexDegreeSampler m_vertexDegreeSampler;
public:
    SingleEdgeDegreeProposer(double shift=1):
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }
};


} // namespace FastMIDyNet


#endif
