#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"

using namespace std;
using namespace FastMIDyNet;
using namespace BaseGraph;

void StochasticBlockModelFamily::sample(){
    m_blockPriorPtr->sample();
    m_edgeMatrixPriorPtr->sample();
    setGraph(generateSBM(m_blockPriorPtr->getState(), m_edgeMatrixPriorPtr->getState().getAdjacencyMatrix()));
    computationFinished();
}


MultiGraph StochasticBlockModelFamily::getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
    size_t numBlocks = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
    MultiGraph edgeMat(numBlocks);
    for (auto idx : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(idx)){
            if (idx > neighbor.vertexIndex)
                continue;
            BlockIndex r = blockSeq[idx], s = blockSeq[neighbor.vertexIndex];
            edgeMat.addMultiedgeIdx(r, s, neighbor.label);
        }
    }
    return edgeMat;
};

void StochasticBlockModelFamily::checkGraphConsistencyWithEdgeMatrix(
    const MultiGraph& graph,
    const BlockSequence& blockSeq,
    const MultiGraph& expectedEdgeMat){
    MultiGraph actualEdgeMat = getEdgeMatrixFromGraph(graph, blockSeq);
    for (auto r : actualEdgeMat)
        for (auto s : actualEdgeMat.getNeighboursOfIdx(r))
            if (expectedEdgeMat.getEdgeMultiplicityIdx(r, s.vertexIndex) != s.label)
                throw ConsistencyError("StochasticBlockModelFamily: at indices ("
                + to_string(r) + ", " + to_string(s.vertexIndex) + ") edge matrix is inconsistent with graph:"
                + to_string(expectedEdgeMat.getEdgeMultiplicityIdx(r, s.vertexIndex)) + " != "
                + to_string(s.label));


};

void StochasticBlockModelFamily::checkSelfConsistency() const{
    m_blockPriorPtr->checkSelfConsistency();
    m_edgeMatrixPriorPtr->checkSelfConsistency();

    checkGraphConsistencyWithEdgeMatrix(m_graph, getLabels(), m_edgeMatrixPriorPtr->getState());
}

void StochasticBlockModelFamily::checkSelfSafety()const{
    if (m_blockPriorPtr == nullptr)
        throw SafetyError("StochasticBlockModelFamily: unsafe family since `m_blockPriorPtr` is empty.");
    m_blockPriorPtr->checkSafety();

    if (m_edgeMatrixPriorPtr == nullptr)
        throw SafetyError("StochasticBlockModelFamily: unsafe family since `m_edgeMatrixPriorPtr` is empty.");
    m_edgeMatrixPriorPtr->checkSafety();
}
