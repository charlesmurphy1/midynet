#ifndef FAST_MIDYNET_VERTEX_SAMPLER_H
#define FAST_MIDYNET_VERTEX_SAMPLER_H

#include <random>

#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

#include "BaseGraph/types.h"

#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class VertexSampler{
protected:
    bool m_withIsolatedVertices = true;
public:
    virtual BaseGraph::VertexIndex sample() const = 0;
    virtual void setUp(const MultiGraph&) = 0;
    virtual void update(const GraphMove&) { };
    virtual void update(const BlockMove&) { };
    virtual double getLogProposalProbRatio(const GraphMove&) const = 0;

    bool setAcceptIsolated(bool accept) { return m_withIsolatedVertices = accept; }
    bool getAcceptIsolated() const { return m_withIsolatedVertices; }
};

class VertexUniformSampler: public VertexSampler{
private:
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler = sset::SamplableSet<BaseGraph::VertexIndex>(1, 100);
public:
    VertexUniformSampler(){}
    VertexUniformSampler(const VertexUniformSampler& other):
        m_vertexSampler(other.m_vertexSampler){ m_withIsolatedVertices = other.m_withIsolatedVertices; }
    virtual ~VertexUniformSampler() {}
    const VertexUniformSampler& operator=(const VertexUniformSampler& other){
        m_vertexSampler = other.m_vertexSampler;
        m_withIsolatedVertices = other.m_withIsolatedVertices;
        return *this;
    }

    BaseGraph::VertexIndex sample() const override { return m_vertexSampler.sample_ext_RNG(rng).first; }
    void setUp(const MultiGraph& graph) override;
    double getLogProposalProbRatio(const GraphMove&) const override { return 0.; }

};

class VertexDegreeSampler: public VertexSampler{
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSampler = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    mutable std::bernoulli_distribution m_vertexChoiceDistribution = std::bernoulli_distribution(.5);
    double m_shift;
public:
    VertexDegreeSampler(double shift=1):m_shift(shift){};
    VertexDegreeSampler(const VertexDegreeSampler& other):
        m_edgeSampler(other.m_edgeSampler){ m_withIsolatedVertices = other.m_withIsolatedVertices; }
    ~VertexDegreeSampler() {}
    const VertexDegreeSampler& operator=(const VertexDegreeSampler& other){
        this->m_edgeSampler = other.m_edgeSampler;
        this->m_withIsolatedVertices = other.m_withIsolatedVertices;
        return *this;
    }

    BaseGraph::VertexIndex sample() const override;
    void setUp(const MultiGraph& graph) override;
    void update(const GraphMove& move) override;
    double getLogProposalProbRatio(const GraphMove&) const override { return 0.; }
};

}/* FastMIDyNet */

#endif
