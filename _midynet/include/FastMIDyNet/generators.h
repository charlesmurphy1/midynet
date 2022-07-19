#ifndef FAST_MIDYNET_GENERATORS_H
#define FAST_MIDYNET_GENERATORS_H

#include <random>
#include <vector>
#include <list>

#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/algorithms/randomgraphs.h"
#include "BaseGraph/types.h"

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/rng.h"


namespace FastMIDyNet{


template < typename InType, typename OutType >
OutType generateCategorical(const std::vector<InType>& probs){
    std::discrete_distribution<OutType> dist(probs.begin(), probs.end());
    return dist(rng);

};
std::vector<size_t> sampleUniformlySequenceWithoutReplacement(size_t n, size_t k);
std::list<size_t> sampleRandomComposition(size_t n, size_t k);
std::list<size_t> sampleRandomWeakComposition(size_t n, size_t k);
std::list<size_t> sampleRandomRestrictedPartition(size_t n, size_t k, size_t numberOfSteps=0);
std::vector<size_t> sampleRandomPermutation(const std::vector<size_t>& nk);

template <typename Iterator>
Iterator sampleUniformlyFrom(Iterator start, Iterator end) {
    std::uniform_int_distribution<> dist(0, std::distance(start, end) - 1);
    std::advance(start, dist(rng));
    return start;
}

BaseGraph::UndirectedMultigraph generateDCSBM(const BlockSequence& vertexBlocks,
        const EdgeMatrix& blockEdgeMatrix, const DegreeSequence& degrees);
BaseGraph::UndirectedMultigraph generateStubLabeledSBM(const BlockSequence& vertexBlocks, const EdgeMatrix& blockEdgeMatrix, bool withSelfLoops=true);
BaseGraph::UndirectedMultigraph generateMultiGraphSBM(const BlockSequence& vertexBlocks, const EdgeMatrix& blockEdgeMatrix, bool withSelfLoops=true);
MultiGraph generateCM(const DegreeSequence& degrees);

MultiGraph generateErdosRenyi(size_t size, size_t edgeCount, bool withSelfLoops=true);
MultiGraph generateStubLabeledErdosRenyi(size_t size, size_t edgeCount, bool withSelfLoops=true);
MultiGraph generateMultiGraphErdosRenyi(size_t size, size_t edgeCount, bool withSelfLoops=true);

template<typename T>
T pickElementUniformly(const std::vector<T>& sequence) {
    return sequence[std::uniform_int_distribution<size_t>(0, sequence.size()-1)(rng)];
}

} // namespace FastMIDyNet

#endif
