#include <fstream>

#include "BaseGraph/fileio.h"
#include "FastMIDyNet/mcmc/callbacks/collector.h"

namespace FastMIDyNet{

void WriteGraphToFileOnSweep::collect(){
    std::ofstream file;
    file.open(m_filename + "_" + std::to_string(m_mcmcPtr->getNumSweeps()) + m_ext);

    // BaseGraph::writeEdgeListIdxInBinaryFile(m_mcmcPtr->getGraph(), file);
    // BaseGraph::writeEdgeListInBinaryFile(m_mcmcPtr->getGraph(), file);

    file.close();
}

void CollectEdgeMultiplicityOnSweep::collect(){
    const MultiGraph& graph = m_mcmcPtr->getGraph();

    for ( auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            if (neighbor.vertexIndex > idx){
                m_edgeMultiplicity.addMultiedgeIdx(idx, neighbor.vertexIndex, neighbor.label, true);
            }
        }
    }
}

} // FastMIDyNet
