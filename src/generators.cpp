#include <random>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>

#include "BaseGraph/types.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


int generateCategorical(const std::vector<double>& probs, RNG& rng){
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

std::list<int> sampleUniformlySequenceWithoutReplacement(size_t n, size_t k, RNG& rng) {
    std::unordered_map<size_t, size_t> indexReplacements;
    size_t newDrawnIndex;
    std::list<int> drawnIndices;

    for (size_t i=0; i<k; i++) {
        newDrawnIndex = std::uniform_int_distribution<size_t>(i, n)(rng);

        if (indexReplacements.find(newDrawnIndex) == indexReplacements.end())
            drawnIndices.push_back(newDrawnIndex);
        else
            drawnIndices.push_back(indexReplacements[newDrawnIndex]);

        if (indexReplacements.find(i) == indexReplacements.end())
            indexReplacements[newDrawnIndex] = i;
        else
            indexReplacements[newDrawnIndex] = indexReplacements[i];
    }
    return drawnIndices;
}

BaseGraph::UndirectedMultigraph generateDCSBM(const std::vector<size_t>& vertexBlocks,
        const Matrix<size_t>& blockEdgeMatrix, const std::vector<size_t>& degrees, RNG& rng) {
    if (degrees.size() != vertexBlocks.size())
        throw std::logic_error("generateDCSBM: Degrees don't have the same length as vertexBlocks.");
    if (*std::max(vertexBlocks.begin(), vertexBlocks.end()) >= blockEdgeMatrix.size())
        throw std::logic_error("generateDCSBM: Vertex is out of range of blockEdgeMatrix.");

    size_t vertexNumber = degrees.size();
    size_t blockNumber = blockEdgeMatrix.size();

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[vertexBlocks[vertex]].push_back(vertex);

    std::vector<std::vector<size_t>> stubsOfBlock(blockNumber);
    for (size_t block=0; block<blockNumber; block++) {
        size_t sumEdgeMatrix(0);

        for (size_t otherBlock=0; otherBlock<blockNumber; otherBlock++)
            sumEdgeMatrix += blockEdgeMatrix[block][otherBlock];

        for (auto vertex: verticesInBlock[block])
            stubsOfBlock[block].insert(stubsOfBlock[block].end(), degrees[vertex], vertex);

        if (stubsOfBlock[block].size() != sumEdgeMatrix)
            throw std::logic_error("generateDCSBM: Edge matrix doesn't match with degrees. "
                    "Sum of row doesn't equal the sum of nodes in block "+std::to_string(block)+".");

        std::random_shuffle(stubsOfBlock[block].begin(), stubsOfBlock[block].end());
    }

    FastMIDyNet::MultiGraph multigraph(vertexNumber);

    size_t edgeNumberBetweenBlocks;
    size_t vertex1, vertex2;
    for (size_t inBlock=0; inBlock<blockNumber; inBlock++) {
        for (size_t outBlock=inBlock; outBlock<blockNumber; outBlock++) {
            edgeNumberBetweenBlocks = blockEdgeMatrix[inBlock][outBlock];
            if (inBlock==outBlock)
                edgeNumberBetweenBlocks /= 2;

            for (size_t edge=0; edge<edgeNumberBetweenBlocks; edge++) {
                vertex1 = *--stubsOfBlock[inBlock].end();
                stubsOfBlock[inBlock].pop_back();
                vertex2 = *--stubsOfBlock[outBlock].end();
                stubsOfBlock[outBlock].pop_back();

                multigraph.addEdgeIdx(vertex1, vertex2);
            }
        }
    }
    return multigraph;
}

BaseGraph::UndirectedMultigraph generateSBM(const std::vector<size_t>& vertexBlocks,
        const Matrix<size_t>& blockEdgeMatrix, RNG& rng) {
    if (*std::max(vertexBlocks.begin(), vertexBlocks.end()) >= blockEdgeMatrix.size())
        throw std::logic_error("generateSBM: Vertex is out of range of blockEdgeMatrix.");

    size_t vertexNumber = vertexBlocks.size();
    size_t blockNumber = blockEdgeMatrix.size();

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[vertexBlocks[vertex]].push_back(vertex);

    FastMIDyNet::MultiGraph multigraph(vertexNumber);

    size_t edgeNumberBetweenBlocks;
    size_t vertex1, vertex2;
    for (size_t inBlock=0; inBlock!=blockNumber; inBlock++) {
        for (size_t outBlock=inBlock; outBlock!=blockNumber; outBlock++) {
            edgeNumberBetweenBlocks = blockEdgeMatrix[inBlock][outBlock];
            if (inBlock==outBlock)
                edgeNumberBetweenBlocks /= 2;

            for (size_t edge=0; edge<edgeNumberBetweenBlocks; edge++) {
                vertex1 = pickElementUniformly<size_t>(verticesInBlock[outBlock], rng);
                vertex2 = pickElementUniformly<size_t>(verticesInBlock[inBlock], rng);
                multigraph.addEdgeIdx(vertex1, vertex2);
            }
        }
    }
    return multigraph;
}

FastMIDyNet::MultiGraph generateCM(const std::vector<size_t>& degrees) {
    size_t n = degrees.size();
    FastMIDyNet::MultiGraph randomGraph(n);

    std::vector<size_t> stubs;

    for (size_t i=0; i<n; i++){
        const size_t& degree = degrees[i];
        if (degree > 0)
            stubs.insert(stubs.end(), degree, i);
    }

    random_shuffle(stubs.begin(), stubs.end());

    size_t vertex1, vertex2;
    auto stubIterator = stubs.begin();
    while (stubIterator != stubs.end()) {
        vertex1 = *stubIterator++;
        vertex2 = *stubIterator++;
        randomGraph.addEdgeIdx(vertex1, vertex2);
    }

    return randomGraph;
}

} // namespace FastMIDyNet
