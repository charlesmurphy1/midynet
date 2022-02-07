#include "FastMIDyNet/proposer/vertex_sampler.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"

namespace FastMIDyNet{

void VertexUniformSampler::setUp(const MultiGraph& graph){
    clear();
    for (auto vertex : graph){
        m_vertexSampler.insert(vertex, 1.);
    }
}


BaseGraph::VertexIndex VertexDegreeSampler::sample() const {
    if (m_sampleFromUniformDistribution(rng))
        return m_vertexSampler.sample_ext_RNG(rng).first;
    auto edge = m_edgeSampler.sample();

    if ( m_vertexChoiceDistribution(rng) )
        return edge.first;
    else
        return edge.second;
}

void VertexDegreeSampler::setUp(const MultiGraph& graph){
    clear();
    for (auto vertex: graph)
        m_vertexSampler.insert(vertex, 1.);
    m_edgeSampler.setUp(graph);
    m_sampleFromUniformDistribution = std::bernoulli_distribution(
        m_shift / (m_shift * graph.getSize() + graph.getTotalEdgeNumber())
    );
}

}
