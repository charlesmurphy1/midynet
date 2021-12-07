#include <fstream>

#include "BaseGraph/fileio.h"
#include "FastMIDyNet/mcmc/callbacks/collector.h"

namespace FastMIDyNet{

void WriteGraphToFileOnSweep::collect(){
    std::ofstream file;
    file.open(m_filename + "_" + std::to_string(m_dynamicsMCMCPtr->getNumSweeps()) + m_ext);

    // BaseGraph::writeEdgeListIdxInBinaryFile(m_dynamicsMCMCPtr->getGraph(), file);
    // BaseGraph::writeEdgeListInBinaryFile(m_dynamicsMCMCPtr->getGraph(), file);

    file.close();
}

}
