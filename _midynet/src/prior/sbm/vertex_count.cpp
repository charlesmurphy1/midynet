
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/prior/sbm/vertex_count.h"


using namespace std;

namespace FastMIDyNet{

void VertexCountUniformPrior::sampleState(){
    list<size_t> vertexCountList = sampleRandomComposition(getSize(), getBlockCount());
    vector<size_t> vertexCount(getBlockCount(), 0);
    size_t r = 0;
    for (auto nr : vertexCountList){
        vertexCount[r] = nr;
        ++r;
    }

    setState( vertexCount );
}

const double VertexCountUniformPrior::getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const {
    return getLogLikelihoodFromState(getSize(), getBlockCount() + move.addedBlocks) - getLogLikelihood();
}


void VertexCountUniformPrior::checkSelfConsistency() const{
    if (m_state.size() != getBlockCount())
        throw ConsistencyError("VertexCountPrior: state size is different from actual block count: "
        + to_string(m_state.size()) + " != " + to_string(getBlockCount()));

    size_t sum = getSizeFromState(m_state);

    if ( sum != getSize() )
        throw ConsistencyError("VertexCountPrior: state sum is different from actual graph size: "
        + to_string(sum) + " != " + to_string(getSize()));
}

} // FastMIDyNet
