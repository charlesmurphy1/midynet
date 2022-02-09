#ifndef FAST_MIDYNET_HINGE_FLIP_H
#define FAST_MIDYNET_HINGE_FLIP_H


#include "FastMIDyNet/exceptions.h"
#include "edge_proposer.h"
#include "FastMIDyNet/proposer/sampler/vertex_sampler.h"
#include "FastMIDyNet/proposer/sampler/edge_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace FastMIDyNet {

class HingeFlipProposer: public EdgeProposer {
private:
    mutable std::bernoulli_distribution m_flipOrientationDistribution = std::bernoulli_distribution(.5);
protected:
    EdgeSampler m_edgeSampler;
    VertexSampler* m_vertexSamplerPtr = nullptr;
public:
    using EdgeProposer::EdgeProposer;
    GraphMove proposeRawMove() const override;
    void setUpFromGraph(const MultiGraph&) override;
    void setVertexSampler(VertexSampler& vertexSampler){ m_vertexSamplerPtr = &vertexSampler; }
    void applyGraphMove(const GraphMove& move) override;
    void applyBlockMove(const BlockMove& move) override { };
    void checkSafety() const override {
        if (m_vertexSamplerPtr == nullptr)
            throw SafetyError("HingeFlipProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
        m_vertexSamplerPtr->checkSafety();
    }
    void clear() override {
        m_edgeSampler.clear();
        m_vertexSamplerPtr->clear();
    }

};


class HingeFlipUniformProposer: public HingeFlipProposer{
private:
    VertexUniformSampler m_vertexUniformSampler = VertexUniformSampler();
public:
    HingeFlipUniformProposer(bool allowSelfLoops=true, bool allowMultiEdges=true):
        HingeFlipProposer(allowSelfLoops, allowMultiEdges){ m_vertexSamplerPtr = &m_vertexUniformSampler; }
    virtual ~HingeFlipUniformProposer(){}
    const double getLogProposalProbRatio(const GraphMove&) const override{ return 0.; }
    void checkConsistency() const override {
        for (auto vertex : *m_graphPtr)
            if (not m_vertexUniformSampler.contains(vertex))
                throw ConsistencyError(
                    "HingeFlipUniformProposer: vertexSampler is inconsistent with graph, "
                    + std::to_string(vertex) + " is not in sampler."
                );
    }
};


class HingeFlipDegreeProposer: public HingeFlipProposer{
private:
    VertexDegreeSampler m_vertexDegreeSampler;

public:
    HingeFlipDegreeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, double shift=1):
        HingeFlipProposer(allowSelfLoops, allowMultiEdges),
        m_vertexDegreeSampler(shift){ m_vertexSamplerPtr = &m_vertexDegreeSampler; }
    virtual ~HingeFlipDegreeProposer(){}

    const double getLogProposalProbRatio(const GraphMove& move) const override{
        auto commonVertex = move.addedEdges[0].first;
        auto gainingVertex = move.addedEdges[0].second;
        auto losingVertex = move.removedEdges[0].second;
        return log(m_vertexDegreeSampler.getVertexWeight(losingVertex) - 1)
             - log(m_vertexDegreeSampler.getVertexWeight(gainingVertex));
    }

    void checkConsistency() const override { }
};


} // namespace FastMIDyNet


#endif
