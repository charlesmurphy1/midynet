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
    virtual void onVertexInsertion(const BaseGraph::VertexIndex& vertex) = 0;
    virtual void onVertexErasure(const BaseGraph::VertexIndex& vertex) = 0;
    virtual void onEdgeInsertion(const BaseGraph::Edge& edge, double edgeWeight) = 0;
    virtual void onEdgeErasure(const BaseGraph::Edge&) = 0;
    virtual void onEdgeAddition(const BaseGraph::Edge&) = 0;
    virtual void onEdgeRemoval(const BaseGraph::Edge&) = 0;
    virtual const double getVertexWeight(const BaseGraph::VertexIndex&) const = 0;
    virtual const double getTotalWeight() const = 0;
    virtual const size_t getSize() const = 0;
    virtual void checkSafety() const {}
    virtual void clear() {}
    void setUpWithGraph(const MultiGraph& graph) ;
};

class VertexUniformSampler: public VertexSampler{
protected:
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler;
public:
    VertexUniformSampler(double minWeight=1, double maxWeight=100):
        m_vertexSampler(minWeight, maxWeight) { }
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
    void onVertexInsertion(const BaseGraph::VertexIndex& vertex) override {
        if (not contains(vertex))
            m_vertexSampler.insert(vertex, 1);
    };
    void onVertexErasure(const BaseGraph::VertexIndex& vertex) override {
        if (not contains(vertex))
            throw std::logic_error("Cannot remove non-exising vertex " + std::to_string(vertex) + ".");
        m_vertexSampler.erase(vertex);
    };
    void onEdgeInsertion(const BaseGraph::Edge& edge, double edgeWeight) { };
    void onEdgeErasure(const BaseGraph::Edge&) { };
    void onEdgeAddition(const BaseGraph::Edge&) { };
    void onEdgeRemoval(const BaseGraph::Edge&) { };
    const double getVertexWeight(const BaseGraph::VertexIndex& vertex) const override {
        return (contains(vertex)) ? m_vertexSampler.get_weight(vertex) : 0.;
    }
    const double getTotalWeight() const override { return m_vertexSampler.total_weight(); }
    const size_t getSize() const override { return m_vertexSampler.size(); }


    void clear() override { m_vertexSampler.clear(); }

    void checkSafety()const override {
        if (m_vertexSampler.size() == 0)
            throw SafetyError("VertexUniformSampler: unsafe vertex sampler since `m_vertexSampler` is empty.");
    }

};

class VertexDegreeSampler: public VertexSampler{
protected:
    sset::SamplableSet<BaseGraph::VertexIndex> m_vertexSampler;
    EdgeSampler m_edgeSampler;
    mutable std::bernoulli_distribution m_vertexChoiceDistribution = std::bernoulli_distribution(.5);
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);
    double m_shift;
    double m_totalEdgeWeight;
    std::unordered_map<BaseGraph::VertexIndex, double> m_weights;
public:
    VertexDegreeSampler(double shift=1, double minWeight=1, double maxWeight=100):
        m_shift(shift), m_edgeSampler(minWeight, maxWeight),
        m_vertexSampler(minWeight, maxWeight){};
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
    void onVertexInsertion(const BaseGraph::VertexIndex& vertex) override;
    void onVertexErasure(const BaseGraph::VertexIndex& vertex) override;
    void onEdgeInsertion(const BaseGraph::Edge& edge, double edgeWeight) override ;
    void onEdgeErasure(const BaseGraph::Edge& edge) override ;
    void onEdgeAddition(const BaseGraph::Edge& edge) override ;
    void onEdgeRemoval(const BaseGraph::Edge& edge) override ;
    bool contains(const BaseGraph::VertexIndex& vertex) const override{
        return m_vertexSampler.count(vertex) > 0;
    }
    const double getVertexWeight(const BaseGraph::VertexIndex& vertex) const override {
        return (contains(vertex)) ? m_shift + m_weights.at(vertex) : 0.;
    }
    const double getTotalWeight() const override {
        return m_totalEdgeWeight + m_shift * m_vertexSampler.total_weight();
    }
    const size_t getSize() const override { return m_vertexSampler.size(); }
    void clear() override { m_totalEdgeWeight = 0; m_vertexSampler.clear(); m_weights.clear(); m_edgeSampler.clear(); }

    void checkSafety() const override {
        if (m_edgeSampler.getTotalWeight() == 0)
            throw SafetyError("VertexDegreeSampler: unsafe vertex sampler since `m_edgeSampler` is empty.");
    }
};

}/* FastMIDyNet */

#endif
