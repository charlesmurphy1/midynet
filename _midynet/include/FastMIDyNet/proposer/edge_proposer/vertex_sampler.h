#ifndef FAST_MIDYNET_VERTEX_SAMPLER_H
#define FAST_MIDYNET_VERTEX_SAMPLER_H

#include <random>
#include <unordered_set>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "BaseGraph/types.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

class VertexSampler{
public:
    virtual BaseGraph::VertexIndex sample() const = 0;
    virtual void setUp(
        const MultiGraph& graph,
        std::unordered_set<BaseGraph::VertexIndex> blackList={}
    ) = 0;
    virtual void update(const GraphMove&) { };
    virtual void update(const BlockMove&) { };
    virtual const double getVertexWeight(const BaseGraph::VertexIndex&) const = 0;
    virtual const double getTotalWeight() const = 0;

    virtual void checkSafety() const {}
};

class VertexUniformSampler: public VertexSampler{
private:
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler = sset::SamplableSet<BaseGraph::VertexIndex>(1, 100);
public:
    VertexUniformSampler(){}
    VertexUniformSampler(const VertexUniformSampler& other):
        m_vertexSampler(other.m_vertexSampler){ }
    virtual ~VertexUniformSampler() {}
    const VertexUniformSampler& operator=(const VertexUniformSampler& other){
        m_vertexSampler = other.m_vertexSampler;
        return *this;
    }

    BaseGraph::VertexIndex sample() const override { return m_vertexSampler.sample_ext_RNG(rng).first; }
    void setUp(
        const MultiGraph& graph,
        std::unordered_set<BaseGraph::VertexIndex> blackList={}
    ) override;
    const double getVertexWeight(const BaseGraph::VertexIndex& vertexIdx) const override { return 1.; }
    const double getTotalWeight() const override { return m_vertexSampler.total_weight(); }

    void checkSafety()const override {
        if (m_vertexSampler.size() == 0)
            throw SafetyError("VertexUniformSampler: unsafe vertex sampler since `m_vertexSampler` is empty.");
    }

};

class VertexDegreeSampler: public VertexSampler{
private:
    sset::SamplableSet<BaseGraph::Edge> m_edgeSampler = sset::SamplableSet<BaseGraph::Edge> (1, 100);
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler = sset::SamplableSet<BaseGraph::VertexIndex>(1, 100);
    mutable std::bernoulli_distribution m_vertexChoiceDistribution = std::bernoulli_distribution(.5);
    mutable std::bernoulli_distribution m_sampleFromUniformDistribution;
    double m_shift;
    std::vector<size_t> m_degrees;
public:
    VertexDegreeSampler(double shift=1):m_shift(shift){};
    VertexDegreeSampler(const VertexDegreeSampler& other):
        m_edgeSampler(other.m_edgeSampler){}
    ~VertexDegreeSampler() {}
    const VertexDegreeSampler& operator=(const VertexDegreeSampler& other){
        this->m_edgeSampler = other.m_edgeSampler;
        return *this;
    }

    BaseGraph::VertexIndex sample() const override;
    void setUp(
        const MultiGraph& graph,
        std::unordered_set<BaseGraph::VertexIndex> blackList={}
    ) override;
    void update(const GraphMove& move) override;
    const double getVertexWeight(const BaseGraph::VertexIndex& vertexIdx) const override {
        return (m_shift + m_degrees[vertexIdx]) / getTotalWeight();
    }
    const double getTotalWeight() const override { return 2 * m_edgeSampler.total_weight() + m_shift * m_vertexSampler.total_weight(); }
    void checkSafety()const override {
        if (m_degrees.size() == 0)
            throw SafetyError("VertexDegreeSampler: unsafe vertex sampler since `m_degrees` is empty.");
        if (m_edgeSampler.size() == 0)
            throw SafetyError("VertexDegreeSampler: unsafe vertex sampler since `m_edgeSampler` is empty.");
    }
};

}/* FastMIDyNet */

#endif
