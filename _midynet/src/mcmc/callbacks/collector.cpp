#include <fstream>

#include "BaseGraph/fileio.h"
#include "FastMIDyNet/mcmc/callbacks/collector.h"
#include "FastMIDyNet/utility/functions.h"

namespace FastMIDyNet{

void WriteGraphToFileOnSweep::collect(){
    std::ofstream file;
    file.open(m_filename + "_" + std::to_string(m_mcmcPtr->getNumSweeps()) + m_ext);

    // BaseGraph::writeEdgeListIdxInBinaryFile(m_mcmcPtr->getGraph(), file);
    // BaseGraph::writeEdgeListInBinaryFile(m_mcmcPtr->getGraph(), file);

    file.close();
}

void CollectEdgeMultiplicityOnSweep::collect(){
    ++m_totalCount;
    const MultiGraph& graph = m_mcmcPtr->getGraph();

    for ( auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            if (neighbor.vertexIndex > idx){
                auto edge = getOrderedPair<BaseGraph::VertexIndex>({idx, neighbor.vertexIndex});
                m_observedEdges.increment(edge);
                m_observedEdgesCount.increment({edge, neighbor.label});
                if (neighbor.label > m_observedEdgesMaxCount[edge])
                    m_observedEdgesMaxCount.set(edge, neighbor.label);
            }
        }
    }
}

const double CollectEdgeMultiplicityOnSweep::getEdgeCountProb(BaseGraph::Edge edge, size_t count) const {
    if (count == 0)
        return 1 - ((double)m_observedEdges.get(edge)) / ((double)m_totalCount);
    else
        return ((double)m_observedEdgesCount.get({edge, count})) / ((double)m_totalCount);
}

const double CollectEdgeMultiplicityOnSweep::getMarginalEntropy() {
    double marginalEntropy = 0;
    for (auto edge : m_observedEdges){
        for (size_t count = 0; count <= m_observedEdgesMaxCount[edge.first]; ++count){
            double p = getEdgeCountProb(edge.first, count);
            if (p > 0)
                marginalEntropy -= p * log(p);
        }
    }
    return marginalEntropy;
}

} // FastMIDyNet
