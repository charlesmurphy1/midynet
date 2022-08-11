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
#include "FastMIDyNet/proposer/edge/hinge_flip.h"


namespace FastMIDyNet {


// int generateCategorical(const std::vector<double>& probs){
//     std::discrete_distribution<int> dist(probs.begin(), probs.end());
//     return dist(rng);
// }


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
    if (k == 1){
        composition.push_back(n);
        return composition;
    }
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

std::vector<size_t> sampleMultinomial(const size_t n, const std::vector<double>& p){
    std::vector<size_t> output(p.size(), 0);

    std::vector<size_t> idx = argsortVector(p);
    std::vector<double> sorted = sortVector(p);
    double norm = 0;
    for (auto pp : p)
        norm += pp;
    if (norm - 1 > 1e-15)
        throw std::runtime_error(
            "sampleMultinomial: `p` must be normalized, but summed to " + std::to_string(norm) + "."
        );

    std::uniform_real_distribution<double> dist(0, 1);
    for (size_t i = 0; i<n; ++i){
        norm = 0;
        std::set<size_t> s;
        size_t j = 0;
        for(size_t j=0; j<p.size(); ++j){
            norm += sorted[j];
            if (dist(rng) <= norm)
                s.insert(idx[j]);
        }
        ++output[(s.size() != p.size()) ? *s.rend() : 0];
    }
    return output;
}
std::vector<size_t> sampleUniformMultinomial(const size_t n, const size_t k){
    std::vector<double> p;
    for (size_t i=0; i<k; ++i)
        p.push_back(1./k);
    return sampleMultinomial(n, p);
}



BaseGraph::UndirectedMultigraph generateDCSBM(
    const BlockSequence& blockSeq,
    const MultiGraph& labelGraph,
    const DegreeSequence& degrees) {

    if (degrees.size() != blockSeq.size())
        throw std::logic_error("generateDCSBM: Degrees don't have the same length as blockSeq.");
    if (*std::max_element(blockSeq.begin(), blockSeq.end()) >= labelGraph.getSize())
        throw std::logic_error("generateDCSBM: Vertex is out of range of labelGraph.");

    size_t vertexNumber = degrees.size();
    size_t blockNumber = labelGraph.getSize();

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[blockSeq[vertex]].push_back(vertex);

    std::vector<std::vector<size_t>> stubsOfBlock(blockNumber);
    for (size_t block=0; block<blockNumber; block++) {
        size_t sumEdgeMatrix(0);

        for (size_t otherBlock=0; otherBlock<blockNumber; otherBlock++)
            sumEdgeMatrix += ((block == otherBlock) ? 2 : 1) * labelGraph.getEdgeMultiplicityIdx(block, otherBlock);

        for (auto vertex: verticesInBlock[block])
            stubsOfBlock[block].insert(stubsOfBlock[block].end(), degrees[vertex], vertex);

        if (stubsOfBlock[block].size() != sumEdgeMatrix)
            throw std::logic_error("generateDCSBM: Edge matrix doesn't match with degrees. "
                    "Sum of row doesn't equal the sum of nodes in block "+std::to_string(block)+".");

        std::random_shuffle(stubsOfBlock[block].begin(), stubsOfBlock[block].end());
    }

    MultiGraph multigraph(vertexNumber);

    size_t edgeNumberBetweenBlocks;
    size_t vertex1, vertex2;
    for (size_t inBlock=0; inBlock<blockNumber; inBlock++) {
        for (size_t outBlock=inBlock; outBlock<blockNumber; outBlock++) {
            edgeNumberBetweenBlocks = labelGraph.getEdgeMultiplicityIdx(inBlock, outBlock);
            // if (inBlock==outBlock)
            //     edgeNumberBetweenBlocks /= 2;

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

BaseGraph::UndirectedMultigraph generateSBM(const BlockSequence& blockSeq, const MultiGraph& labelGraph, bool withSelfLoops) {

    if (*std::max_element(blockSeq.begin(), blockSeq.end()) >= labelGraph.getSize())
        throw std::logic_error("generateSBM: Vertex is out of range of labelGraph.");

    size_t size = blockSeq.size();
    size_t blockCount = labelGraph.getSize();

    std::map<std::pair<BlockIndex, BlockIndex>, std::vector<BaseGraph::Edge>> allLabeledEdges;

    for (size_t i=0; i<size; ++i){
        for (size_t j=0; j<size; ++j){
            if (i > j or (i == j and !withSelfLoops))
                continue;
            auto rs = getOrderedPair<BlockIndex>({blockSeq[i],blockSeq[j]});
            allLabeledEdges[rs].push_back({i, j});
        }
    }

    MultiGraph graph(size);
    for(const auto& labeledEdges : allLabeledEdges){
        BlockIndex r = labeledEdges.first.first, s = labeledEdges.first.second;
        size_t ers = labelGraph.getEdgeMultiplicityIdx(r, s);

        if (labeledEdges.second.size() < ers)
            throw std::logic_error("generateSBM: edge count at r=" + std::to_string(r)
                + " and s=" + std::to_string(s) + " (ers=" + std::to_string(ers)
                + ") must be greater than the total number of pairs ("
                + std::to_string(labeledEdges.second.size()) + ").");
        auto indices = sampleUniformlySequenceWithoutReplacement(labeledEdges.second.size(), ers);
        for (const auto& i: indices)
            graph.addEdgeIdx(labeledEdges.second[i].first, labeledEdges.second[i].second);
    }

    return graph;
}

BaseGraph::UndirectedMultigraph generateStubLabeledSBM(const BlockSequence& blockSeq, const MultiGraph& labelGraph, bool withSelfLoops) {

    if (*std::max_element(blockSeq.begin(), blockSeq.end()) >= labelGraph.getSize())
        throw std::logic_error("generateStubLabeledSBM: Vertex is out of range of labelGraph.");

    size_t vertexNumber = blockSeq.size();
    size_t blockNumber = labelGraph.getSize();

    std::vector<std::vector<size_t>> verticesInBlock(blockNumber);
    for (size_t vertex=0; vertex<vertexNumber; vertex++)
        verticesInBlock[blockSeq[vertex]].push_back(vertex);

    MultiGraph multigraph(vertexNumber);

    size_t edgeNumberBetweenBlocks;
    size_t vertex1, vertex2;
    for (size_t inBlock=0; inBlock!=blockNumber; inBlock++) {
        for (size_t outBlock=inBlock; outBlock!=blockNumber; outBlock++) {
            if (verticesInBlock[inBlock].size() == 0 or verticesInBlock[outBlock].size() == 0)
                continue;
            edgeNumberBetweenBlocks = labelGraph.getEdgeMultiplicityIdx(inBlock, outBlock);
            for (size_t edge=0; edge<edgeNumberBetweenBlocks; edge++) {
                if (withSelfLoops or inBlock != outBlock){
                    vertex1 = pickElementUniformly<size_t>(verticesInBlock[outBlock]);
                    vertex2 = pickElementUniformly<size_t>(verticesInBlock[inBlock]);
                } else {
                    auto p = sampleUniformlySequenceWithoutReplacement(verticesInBlock[inBlock].size(), 2);
                    vertex1 = p[0];
                    vertex2 = p[1];
                }
                multigraph.addEdgeIdx(vertex1, vertex2);
            }
        }
    }
    return multigraph;
}

BaseGraph::UndirectedMultigraph generateMultiGraphSBM(const BlockSequence& blockSeq, const MultiGraph& labelGraph, bool withSelfLoops) {

    // displayVector(blockSeq, "[generator] b", true);
    // displayMatrix(labelGraph, "[generator] E", true);
    if (*std::max_element(blockSeq.begin(), blockSeq.end()) >= labelGraph.getSize())
        throw std::logic_error("generateSBM: Vertex is out of range of edgeMat.");

    size_t size = blockSeq.size();
    size_t blockCount = labelGraph.getSize();

    std::map<std::pair<BlockIndex, BlockIndex>, std::vector<BaseGraph::Edge>> allLabeledEdges;

    for (size_t i=0; i<size; ++i){
        for (size_t j=0; j<size; ++j){
            if (i > j or (i == j and !withSelfLoops))
                continue;
            auto rs = getOrderedPair<BlockIndex>({blockSeq[i],blockSeq[j]});
            allLabeledEdges[rs].push_back({i, j});
        }
    }

    MultiGraph graph(size);
    for(const auto& labeledEdges : allLabeledEdges){
        BlockIndex r = labeledEdges.first.first, s = labeledEdges.first.second;
        size_t ers = labelGraph.getEdgeMultiplicityIdx(r, s);
        auto flatMultiplicity = sampleRandomWeakComposition(ers, labeledEdges.second.size());
        size_t counter = 0;
        for (const auto& m: flatMultiplicity){
            if (m != 0)
                graph.addMultiedgeIdx(labeledEdges.second[counter].first, labeledEdges.second[counter].second, m);
            ++counter;
        }
    }

    return graph;
}

MultiGraph generateCM(const DegreeSequence& degrees) {
    size_t n = degrees.size();
    MultiGraph randomGraph(n);

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

MultiGraph generateErdosRenyi(size_t size, size_t edgeCount, bool withSelfLoops){
    std::vector<BaseGraph::Edge> allEdges;
    for (size_t i=0; i < size; ++i )
        for (size_t j=i; j < size; ++j )
            if (withSelfLoops or j != i)
                allEdges.push_back({i, j});
    if (allEdges.size() < edgeCount)
        throw std::logic_error("generateErdosRenyi: edge count (" + std::to_string(edgeCount) +
            ") must be greater than the total number of pairs (with N=" + std::to_string(size) + ").");
    auto indices = sampleUniformlySequenceWithoutReplacement(allEdges.size(), edgeCount);

    MultiGraph graph(size);
    for (auto i : indices)
        graph.addEdgeIdx(allEdges[i]);
    return graph;
}

MultiGraph generateStubLabeledErdosRenyi(size_t size, size_t edgeCount, bool withSelfLoops){
    MultiGraph graph(size);
    std::uniform_int_distribution<size_t> dist(0, size-1);
    for (size_t e=0; e<edgeCount; ++e){
        BaseGraph::VertexIndex i, j;
        if (withSelfLoops){
            i = dist(rng);
            j = dist(rng);
        }else{
            auto edge = sampleUniformlySequenceWithoutReplacement(size, 2);
            i = edge[0], j = edge[1];
        }
        graph.addMultiedgeIdx(i, j, 1);
    }
    return graph;
}

MultiGraph generateMultiGraphErdosRenyi(size_t size, size_t edgeCount, bool withSelfLoops){
    std::vector<BaseGraph::Edge> allEdges;
    for (size_t i=0; i < size; ++i )
        for (size_t j=i; j < size; ++j )
            if (withSelfLoops or j != i)
                allEdges.push_back({i, j});
    auto flatMultiplicity = sampleRandomWeakComposition(edgeCount, allEdges.size());
    MultiGraph graph(size);
    size_t counter = 0;
    for (auto m: flatMultiplicity){
        if (m != 0)
            graph.addMultiedgeIdx(allEdges[counter].first, allEdges[counter].second, m);
        ++counter;
    }

    return graph;
}

} // namespace FastMIDyNet
