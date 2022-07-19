#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/random_graph/util.h"


namespace FastMIDyNet{

MultiGraph getEdgeMatrixFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq){
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

void checkGraphConsistencyWithEdgeMatrix(
    std::string namePrefix,
    const MultiGraph& graph,
    const BlockSequence& blockSeq,
    const MultiGraph& expectedEdgeMat){
    MultiGraph actualEdgeMat = getEdgeMatrixFromGraph(graph, blockSeq);
    for (auto r : actualEdgeMat)
        for (auto s : actualEdgeMat.getNeighboursOfIdx(r))
            if (expectedEdgeMat.getEdgeMultiplicityIdx(r, s.vertexIndex) != s.label)
                throw ConsistencyError(namePrefix + ": at indices ("
                + std::to_string(r) + ", " + std::to_string(s.vertexIndex) + ") edge matrix is inconsistent with graph:"
                + std::to_string(expectedEdgeMat.getEdgeMultiplicityIdx(r, s.vertexIndex)) + " != "
                + std::to_string(s.label));
};

void checkGraphConsistencyWithDegreeSequence(std::string namePrefix, const MultiGraph& graph, const DegreeSequence& expectedDegreeSeq){
    DegreeSequence actualDegreeSeq = graph.getDegrees();

    for (auto idx : graph){
        if (expectedDegreeSeq[idx] != actualDegreeSeq[idx])
            throw ConsistencyError(namePrefix + ": expected degree of index " + std::to_string(idx)
            + " is inconsistent with graph : " + std::to_string(expectedDegreeSeq[idx]) + " != " + std::to_string(actualDegreeSeq[idx]));
    }
}
}
