#ifndef FAST_MIDYNET_LABELPAIR_SAMPLER_H
#define FAST_MIDYNET_LABELPAIR_SAMPLER_H

#include "edge_sampler.h"
#include "vertex_sampler.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet{

using LabelPair = std::pair<BlockIndex, BlockIndex>;

class LabelPairSampler{
protected:
    double m_shift;
    VertexUniformSampler m_vertexSampler;
    EdgeSampler m_edgeSampler;
    const std::vector<BlockIndex>* m_labelsPtr = nullptr;
    const std::vector<size_t>* m_vertexCountsPtr = nullptr;
    const Matrix<size_t>* m_edgeMatrixPtr = nullptr;
    mutable std::bernoulli_distribution m_bernoulliDistribution = std::bernoulli_distribution(0.5);


public:
    LabelPairSampler(double shift=1): m_shift(shift){}

    LabelPair sample() const ;
    void setUp(const RandomGraph& randomGraph) {
        clear();
        m_labelsPtr = &randomGraph.getBlocks();
        m_vertexCountsPtr = &randomGraph.getVertexCountsInBlocks();
        m_edgeMatrixPtr = &randomGraph.getEdgeMatrix();
        for (auto vertex : randomGraph.getGraph()){
            m_vertexSampler.onVertexInsertion(vertex);
            for (auto neighbor : randomGraph.getGraph().getNeighboursOfIdx(vertex)){
                if (vertex <= neighbor.vertexIndex){
                    m_vertexSampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
                    m_edgeSampler.onEdgeInsertion({vertex, neighbor.vertexIndex}, neighbor.label);
                }
            }
        }
    }
    void onEdgeAddition(const BaseGraph::Edge& edge) {
        m_vertexSampler.onEdgeAddition(edge); m_edgeSampler.onEdgeAddition(edge);
    }
    void onEdgeRemoval(const BaseGraph::Edge& edge) {
        m_vertexSampler.onEdgeRemoval(edge); m_edgeSampler.onEdgeRemoval(edge);
    }
    void onEdgeInsertion(const BaseGraph::Edge& edge, double weight) {
        m_vertexSampler.onEdgeInsertion(edge, weight); m_edgeSampler.onEdgeInsertion(edge, weight);
    }
    void onEdgeErasure(const BaseGraph::Edge& edge) {
        m_vertexSampler.onEdgeErasure(edge);
        m_edgeSampler.onEdgeErasure(edge);
    }

    const double getLabelPairWeight(const LabelPair& pair) const {
        return m_shift * (*m_vertexCountsPtr)[pair.first] * (*m_vertexCountsPtr)[pair.second] + (*m_edgeMatrixPtr)[pair.first][pair.second];
    }
    const double getVertexTotalWeight() const {
        return m_vertexSampler.getTotalWeight();
    }
    const double getEdgeTotalWeight() const { return m_edgeSampler.getTotalWeight(); }
    const double getTotalWeight() const { return m_shift * getVertexTotalWeight() * getVertexTotalWeight() + getEdgeTotalWeight(); }
    const BlockIndex getLabelOfIdx(const BaseGraph::VertexIndex& vertex) const {
        return (*m_labelsPtr)[vertex];
    }
    const LabelPair getLabelOfIdx(const BaseGraph::Edge& edge) const {
        return getOrderedPair<BlockIndex>({getLabelOfIdx(edge.first), getLabelOfIdx(edge.second)});
    }

    void clear() {
        m_labelsPtr = nullptr;
        m_vertexCountsPtr = nullptr;
        m_edgeMatrixPtr = nullptr;
        m_vertexSampler.clear();
        m_edgeSampler.clear();
    }
    void checkSafety() const {
        if (m_labelsPtr == nullptr)
            throw std::logic_error("LabeledEdgeProposer: `m_labelsPtr` is null.");
    }

};

}
#endif
