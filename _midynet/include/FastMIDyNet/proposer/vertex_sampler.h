#ifndef FAST_MIDYNET_VERTEX_SAMPLER_H
#define FAST_MIDYNET_VERTEX_SAMPLER_H

#include <random>
#include <unordered_map>
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "BaseGraph/types.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/rng.h"
#include "edge_sampler.h"

namespace FastMIDyNet{

class VertexSampler{
public:
    virtual BaseGraph::VertexIndex sample() const = 0;
    virtual bool contains(const BaseGraph::VertexIndex&) const = 0;
    virtual void insertVertex(const BaseGraph::VertexIndex& vertex) = 0;
    virtual void insertEdge(const BaseGraph::Edge& edge, double edgeWeight) = 0;
    virtual void eraseEdge(const BaseGraph::Edge&) = 0;
    virtual void addEdge(const BaseGraph::Edge&) = 0;
    virtual void removeEdge(const BaseGraph::Edge&) = 0;
    virtual const double getVertexWeight(const BaseGraph::VertexIndex&) const = 0;
    virtual const double getTotalWeight() const = 0;
    virtual const size_t getSize() const = 0;
    virtual void checkSafety() const {}
    virtual void clear() {}
};

class VertexUniformSampler: public VertexSampler{
protected:
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

    bool contains(const BaseGraph::VertexIndex& vertex) const override {
        return m_vertexSampler.count(vertex) > 0;
    };
    void insertVertex(const BaseGraph::VertexIndex& vertex) override {
        if (not contains(vertex))
            m_vertexSampler.insert(vertex, 1);
    };
    void insertEdge(const BaseGraph::Edge& edge, double edgeWeight) { };
    void eraseEdge(const BaseGraph::Edge&) { };
    void addEdge(const BaseGraph::Edge&) { };
    void removeEdge(const BaseGraph::Edge&) { };
    const double getVertexWeight(const BaseGraph::VertexIndex& vertex) const override {
        return (contains(vertex)) ? m_vertexSampler.get_weight(vertex) : 0.;
    }
    const double getTotalWeight() const override { return m_vertexSampler.total_weight(); }
    const size_t getSize() const override { return m_vertexSampler.size(); }


    void clear() { m_vertexSampler.clear(); }

    void checkSafety()const override {
        if (m_vertexSampler.size() == 0)
            throw SafetyError("VertexUniformSampler: unsafe vertex sampler since `m_vertexSampler` is empty.");
    }

};

class VertexDegreeSampler: public VertexSampler{
protected:
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler = sset::SamplableSet<BaseGraph::VertexIndex>(1, 100);
    EdgeSampler m_edgeSampler;
    mutable std::bernoulli_distribution m_vertexChoiceDistribution = std::bernoulli_distribution(.5);
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);
    double m_shift;
    double m_totalEdgeWeight;
    std::unordered_map<BaseGraph::VertexIndex, double> m_weights;
public:
    VertexDegreeSampler(double shift=1):m_shift(shift){};
    VertexDegreeSampler(const VertexDegreeSampler& other):
        m_vertexSampler(other.m_vertexSampler), m_edgeSampler(other.m_edgeSampler),
        m_totalEdgeWeight(other.m_totalEdgeWeight), m_shift(other.m_shift){}
    ~VertexDegreeSampler() {}
    const VertexDegreeSampler& operator=(const VertexDegreeSampler& other){
        this->m_vertexSampler = other.m_vertexSampler;
        this->m_edgeSampler = other.m_edgeSampler;
        this->m_totalEdgeWeight = other.m_totalEdgeWeight;
        this->m_weights = other.m_weights;
        this->m_shift = other.m_shift;
        return *this;
    }

    BaseGraph::VertexIndex sample() const override;
    void insertVertex(const BaseGraph::VertexIndex& vertex) override;
    void insertEdge(const BaseGraph::Edge& edge, double edgeWeight) override ;
    void eraseEdge(const BaseGraph::Edge& edge) override ;
    void addEdge(const BaseGraph::Edge& edge) override ;
    void removeEdge(const BaseGraph::Edge& edge) override ;
    bool contains(const BaseGraph::VertexIndex& vertex) const override{
        return m_vertexSampler.count(vertex) > 0;
    }
    const double getVertexWeight(const BaseGraph::VertexIndex& vertex) const override {
        return (contains(vertex)) ? m_shift + m_weights.at(vertex) : 0.;
    }
    void applyGraphMove(const GraphMove&) ;
    const double getTotalWeight() const override {
        return m_totalEdgeWeight + m_shift * m_vertexSampler.total_weight();
    }
    const size_t getSize() const override { return m_vertexSampler.size(); }
    void clear() { m_totalEdgeWeight = 0; m_vertexSampler.clear(); m_weights.clear(); m_edgeSampler.clear(); }

    void checkSafety() const override {
        if (m_edgeSampler.getTotalWeight() == 0)
            throw SafetyError("VertexDegreeSampler: unsafe vertex sampler since `m_edgeSampler` is empty.");
    }
};

}/* FastMIDyNet */

#endif
