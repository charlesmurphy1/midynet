#include <random>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>

#include "BaseGraph/types.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {


int generateCategorical(const std::vector<double>& probs){
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}


std::vector<size_t> sampleUniformlySequenceWithoutReplacement(size_t n, size_t k) {
    std::unordered_map<size_t, size_t> indexReplacements;
    size_t newDrawnIndex;
    std::vector<size_t> drawnIndices;

    for (size_t i=0; i<k; i++) {
        newDrawnIndex = std::uniform_int_distribution<size_t>(i, n-1)(rng);

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


std::list<size_t> sampleRandomComposition(size_t n, size_t k) {
    // sample the composition of n into exactly k parts
    std::list<size_t> composition;
    std::vector<size_t> uniformRandomSequence(k-1);

    uniformRandomSequence = sampleUniformlySequenceWithoutReplacement(n-1, k-1);
    std::sort(uniformRandomSequence.begin(), uniformRandomSequence.end());

    composition.push_back(uniformRandomSequence[0] + 1);
    for (size_t i=1; i<uniformRandomSequence.size(); i++)
        composition.push_back(uniformRandomSequence[i] - uniformRandomSequence[i-1]);
    composition.push_back(n - uniformRandomSequence[k-2] - 1);
    return composition;
}


std::list<size_t> sampleRandomWeakComposition(size_t n, size_t k) {
    // sample the weak composition of n into exactly k parts
    if (k == 1){
        std::list<size_t> ret = {n};
        return ret;
    } else if (k == 0){
        std::list<size_t> ret;
        return ret;
    }
    std::list<size_t> weakComposition;
    std::vector<size_t> uniformRandomSequence(k-1);

    uniformRandomSequence = sampleUniformlySequenceWithoutReplacement(n+k-1, k-1);
    std::sort(uniformRandomSequence.begin(), uniformRandomSequence.end());

    weakComposition.push_back(uniformRandomSequence[0]);
    for (size_t i=1; i<uniformRandomSequence.size(); i++)
        weakComposition.push_back(uniformRandomSequence[i] - uniformRandomSequence[i-1] - 1);
    weakComposition.push_back(n + k - 2 - uniformRandomSequence[k-2]);

    return weakComposition;
}


std::list<size_t> sampleRandomRestrictedPartition(size_t n, size_t k, size_t numberOfSteps) {
    // sample the partition of n into exactly k parts with zeros
    if (numberOfSteps==0)
        numberOfSteps = n;

    auto partition = sampleRandomWeakComposition(n, k);
    partition.sort();
    auto skimmedPartition = partition;
    skimmedPartition.unique();
    double P = logMultinomialCoefficient(skimmedPartition);

    for (size_t i=0; i<numberOfSteps; i++) {
        auto newPartition = sampleRandomWeakComposition(n, k);
        newPartition.sort();
        auto skimmedNewPartition = newPartition;
        skimmedNewPartition.unique();
        double Q = logMultinomialCoefficient(skimmedNewPartition);
        if (std::uniform_int_distribution<size_t>(0, 1)(rng) < exp(P - Q)) {
            partition = newPartition;
            P = Q;
        }
    }
    return partition;
}

std::vector<size_t> sampleRandomPermutation(const std::vector<size_t>& nk){
    // sample the permutation of a multiset of K elements with multiciplicity {nk}.
    size_t sum = 0;
    std::vector<size_t> cumul;

    for (auto n : nk){
        sum += n;
        cumul.push_back(sum);
    }

    std::vector<size_t> indices;
    for (size_t i = 0; i < sum; ++i) {
        indices.push_back(i);
    }
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<size_t> sequence(indices.size());
    size_t idx = 0;
    for (size_t i = 0; i < sum; ++i ){
        if (i == cumul[idx]) ++idx;
        sequence[indices[i]] = idx;
    }
    return sequence;
}


BaseGraph::UndirectedMultigraph generateDCSBM(
    const BlockSequence& blockSeq,
    const EdgeMatrix& edgeMat,
    const DegreeSequence& degrees) {

    if (degrees.size() != blockSeq.size())
        throw std::logic_error("generateDCSBM: Degrees don't have the same length as blockSeq.");
    if (*std::max_element(blockSeq.begin(), blockSeq.end()) >= edgeMat.size())
        throw std::logic_error("generateDCSBM: Vertex is out of range of edgeMat.");

    size_t vertexNumber = degrees.size();
    size_t blockNumber = edgeMat.size();

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[blockSeq[vertex]].push_back(vertex);

    std::vector<std::vector<size_t>> stubsOfBlock(blockNumber);
    for (size_t block=0; block<blockNumber; block++) {
        size_t sumEdgeMatrix(0);

        for (size_t otherBlock=0; otherBlock<blockNumber; otherBlock++)
            sumEdgeMatrix += edgeMat[block][otherBlock];

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
            edgeNumberBetweenBlocks = edgeMat[inBlock][outBlock];
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

BaseGraph::UndirectedMultigraph generateSBM(const BlockSequence& blockSeq,
        const EdgeMatrix& edgeMat) {
    if (*std::max_element(blockSeq.begin(), blockSeq.end()) >= edgeMat.size())
        throw std::logic_error("generateSBM: Vertex is out of range of edgeMat.");

    size_t vertexNumber = blockSeq.size();
    size_t blockNumber = edgeMat.size();

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[blockSeq[vertex]].push_back(vertex);

    FastMIDyNet::MultiGraph multigraph(vertexNumber);

    size_t edgeNumberBetweenBlocks;
    size_t vertex1, vertex2;
    for (size_t inBlock=0; inBlock!=blockNumber; inBlock++) {
        for (size_t outBlock=inBlock; outBlock!=blockNumber; outBlock++) {
            edgeNumberBetweenBlocks = edgeMat[inBlock][outBlock];
            if (inBlock==outBlock)
                edgeNumberBetweenBlocks /= 2;

            for (size_t edge=0; edge<edgeNumberBetweenBlocks; edge++) {
                vertex1 = pickElementUniformly<size_t>(verticesInBlock[outBlock]);
                vertex2 = pickElementUniformly<size_t>(verticesInBlock[inBlock]);
                multigraph.addEdgeIdx(vertex1, vertex2);
            }
        }
    }
    return multigraph;
}

FastMIDyNet::MultiGraph generateCM(const DegreeSequence& degrees) {
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
