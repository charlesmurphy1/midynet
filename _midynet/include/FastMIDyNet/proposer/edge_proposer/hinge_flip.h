#ifndef FAST_MIDYNET_HINGE_FLIP_H
#define FAST_MIDYNET_HINGE_FLIP_H


#include "FastMIDyNet/exceptions.h"
#include "edge_proposer.h"
#include "vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

class HingeFlipProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSamplableSet = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    VertexSampler* m_vertexSamplerPtr = NULL;
public:
    using EdgeProposer::EdgeProposer;
    bool setAcceptIsolated(bool accept) override;

    GraphMove proposeMove() const override;
    void setUp(const RandomGraph& randomGraph) override { setUp(randomGraph.getState()); }
    void setUp(const MultiGraph& graph);
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }

    double getLogProposalProbRatio(const GraphMove&) const override{ return 0; }
    void updateProbabilities(const GraphMove& move) override;
    void checkSafety() const override {
        if (m_vertexSamplerPtr == nullptr)
            throw SafetyError("HingeFlipProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
    }
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
