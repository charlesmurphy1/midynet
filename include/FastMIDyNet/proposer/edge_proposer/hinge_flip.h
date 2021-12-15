#ifndef FAST_MIDYNET_HINGE_FLIP_H
#define FAST_MIDYNET_HINGE_FLIP_H


#include "edge_proposer.h"
// #include "FastMIDyNet/proposer/edge_proposer/vertex_sampler.h"
#include "vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

class HingeFlip: public EdgeProposer {
protected:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    // sset::SamplableSet<BaseGraph::VertexIndex> m_nodeSamplableSet = sset::SamplableSet<BaseGraph::VertexIndex> (1, 100);
    VertexSampler* m_vertexSamplerPtr = NULL;
    std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);

public:
    using EdgeProposer::EdgeProposer;
    void acceptIsolated(bool accept);

    GraphMove proposeMove();
    void setUp(const RandomGraph& randomGraph) { setUp(randomGraph.getState()); }
    void setUp(const MultiGraph& graph);

    double getLogProposalProbRatio(const GraphMove&) const { return 0; }
    void updateProbabilities(const GraphMove& move);

    // For tests
    const sset::SamplableSet<BaseGraph::Edge>& getEdgeSamplableSet() { return m_edgeSamplableSet; }
    // const sset::SamplableSet<BaseGraph::VertexIndex>& getNodeSamplableSet() { return m_nodeSamplableSet; }
};

class UniformHingeFlip: public HingeFlip{
private:
    VertexUniformSampler m_vertexUniformSampler = VertexUniformSampler();
public:
    UniformHingeFlip(){
        m_vertexSamplerPtr = &m_vertexUniformSampler;
    }
};

class DegreeHingeFlip: public HingeFlip{
private:
    VertexDegreeSampler m_vertexDegreeSampler;
public:
    DegreeHingeFlip(double shift=1):
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }
};

} // namespace FastMIDyNet


#endif
