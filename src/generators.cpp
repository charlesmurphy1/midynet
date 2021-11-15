#include <random>
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

BaseGraph::UndirectedMultigraph generateDCSBM(const std::vector<size_t> degrees,
            const Matrix<size_t>& blockEdgeMatrix, const std::vector<size_t>& vertexBlocks, RNG& rng) {
    if (degrees.size() != vertexBlocks.size())
        throw std::logic_error("generateDCSBM: Degrees don't have the same length as vertexBlocks.");
    if (*std::max(vertexBlocks.begin(), vertexBlocks.end()) >= blockEdgeMatrix.size()-1)
        throw std::logic_error("generateDCSBM: Vertex is out of range of blockEdgeMatrix.");

    size_t vertexNumber = degrees.size();
    size_t blockNumber = blockEdgeMatrix.size();
    std::vector<size_t> blockDegrees(blockNumber, 0);
    size_t inBlock, outBlock;

    for (inBlock=0; inBlock!=blockNumber; inBlock++) {
        for (outBlock=inBlock+1; outBlock!=blockNumber; outBlock++) {
            blockDegrees[inBlock] += blockEdgeMatrix[inBlock][outBlock];
            blockDegrees[outBlock] += blockEdgeMatrix[outBlock][inBlock];
        }
    }

    std::vector<std::list<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[vertexBlocks[vertex]].push_back(vertex);

    std::vector<size_t> blockStubs(blockNumber);
    std::vector<std::vector<size_t>> vertexStubsOfBlock(blockNumber);

    for (size_t block=0; block!=blockNumber; block++) {
        const auto& blockDegree = blockDegrees[block];
        if (blockDegree>0) {
            blockStubs.insert(blockStubs.end(), blockDegree, block);

            for (auto& vertex: verticesInBlock[block]) {
                const auto& vertexDegree = degrees[vertex];
                if (vertexDegree>0)
                    vertexStubsOfBlock[block].insert(vertexStubsOfBlock[block].end(), vertexDegree, vertex);
            }
            std::random_shuffle(vertexStubsOfBlock[block].begin(), vertexStubsOfBlock[block].end());
        }
    }
    std::random_shuffle(blockStubs.begin(), blockStubs.end());


    FastMIDyNet::MultiGraph multigraph(vertexNumber);

    auto blockStubIterator = blockStubs.begin();
    size_t vertex1, vertex2;
    while (blockStubIterator != blockStubs.end()) {
        inBlock = *blockStubIterator++;
        outBlock = *blockStubIterator++;

        vertex1 = vertexStubsOfBlock[inBlock][vertexStubsOfBlock[inBlock].size()-1];
        vertex2 = vertexStubsOfBlock[outBlock][vertexStubsOfBlock[outBlock].size()-1];

        multigraph.addEdgeIdx(vertex1, vertex2);
        vertexStubsOfBlock[inBlock].pop_back();
        vertexStubsOfBlock[outBlock].pop_back();
    }
    return multigraph;
}

BaseGraph::UndirectedMultigraph generateSBM(const Matrix<size_t>& blockEdgeMatrix, const std::vector<size_t>& vertexBlocks, RNG& rng) {
    if (*std::max(vertexBlocks.begin(), vertexBlocks.end()) >= blockEdgeMatrix.size()-1)
        throw std::logic_error("generateSBM: Vertex is out of range of blockEdgeMatrix.");

    size_t vertexNumber = vertexBlocks.size();
    size_t blockNumber = blockEdgeMatrix.size();
    std::vector<size_t> blockDegrees(blockNumber, 0);
    size_t inBlock, outBlock;

    for (inBlock=0; inBlock!=blockNumber; inBlock++) {
        for (outBlock=inBlock+1; outBlock!=blockNumber; outBlock++) {
            blockDegrees[inBlock] += blockEdgeMatrix[inBlock][outBlock];
            blockDegrees[outBlock] += blockEdgeMatrix[outBlock][inBlock];
        }
    }

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[vertexBlocks[vertex]].push_back(vertex);

    std::vector<size_t> blockStubs(blockNumber);

    for (size_t block=0; block!=blockNumber; block++) {
        const auto& blockDegree = blockDegrees[block];
        if (blockDegree>0)
            blockStubs.insert(blockStubs.end(), blockDegree, block);
    }
    std::random_shuffle(blockStubs.begin(), blockStubs.end());


    FastMIDyNet::MultiGraph multigraph(vertexNumber);

    auto blockStubIterator = blockStubs.begin();
    size_t vertex1, vertex2;
    while (blockStubIterator != blockStubs.end()) {
        inBlock = *blockStubIterator++;
        outBlock = *blockStubIterator++;

        vertex1 = pickElementUniformly<size_t>(verticesInBlock[inBlock], rng);
        vertex2 = pickElementUniformly<size_t>(verticesInBlock[outBlock], rng);
        multigraph.addEdgeIdx(vertex1, vertex2);
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
