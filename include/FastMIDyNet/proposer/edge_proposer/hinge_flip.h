#ifndef FAST_MIDYNET_HINGE_FLIP_H
#define FAST_MIDYNET_HINGE_FLIP_H


#include "edge_proposer.h"
// #include "FastMIDyNet/proposer/edge_proposer/vertex_sampler.h"
#include "vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

class HingeFlipProposer: public EdgeProposer {
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    VertexSampler* m_vertexSamplerPtr = NULL;
public:
    using EdgeProposer::EdgeProposer;
    void acceptIsolated(bool accept);

    GraphMove proposeMove();
    void setUp(const RandomGraph& randomGraph) { setUp(randomGraph.getState()); }
    void setUp(const MultiGraph& graph);
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }

    double getLogProposalProbRatio(const GraphMove&) const { return 0; }
    void updateProbabilities(const GraphMove& move);

    // For tests
    const sset::SamplableSet<BaseGraph::Edge>& getEdgeSamplableSet() { return m_edgeSamplableSet; }
    // const sset::SamplableSet<BaseGraph::VertexIndex>& getNodeSamplableSet() { return m_nodeSamplableSet; }
};

class HingeFlipUniformProposer: public HingeFlipProposer{
private:
    VertexUniformSampler m_vertexUniformSampler = VertexUniformSampler();
public:
    HingeFlipUniformProposer(){
        m_vertexSamplerPtr = &m_vertexUniformSampler;
    }
};

class HingeFlipDegreeProposer: public HingeFlipProposer{
private:
    VertexDegreeSampler m_vertexDegreeSampler;
public:
    HingeFlipDegreeProposer(double shift=1):
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }
};

} // namespace FastMIDyNet


#endif
