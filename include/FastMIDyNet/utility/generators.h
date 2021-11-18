#ifndef FAST_MIDYNET_GENERATORS
#define FAST_MIDYNET_GENERATORS

#include <random>
#include <vector>
#include <list>
#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility.h"


namespace FastMIDyNet{


int generateCategorical(const std::vector<double>& probs);

std::vector<size_t> sampleUniformlySequenceWithoutReplacement(size_t n, size_t k);
std::list<size_t> sampleRandomComposition(size_t n, size_t k);
std::list<size_t> sampleRandomWeakComposition(size_t n, size_t k);

std::list<size_t> sampleRandomRestrictedPartition(size_t n, size_t k, size_t numberOfSteps=0);

BaseGraph::UndirectedMultigraph generateDCSBM(const BlockSequence& vertexBlocks,
        const EdgeMatrix& blockEdgeMatrix, const DegreeSequence& degrees);
BaseGraph::UndirectedMultigraph generateSBM(const BlockSequence& vertexBlocks,
        const EdgeMatrix& blockEdgeMatrix);
FastMIDyNet::MultiGraph generateCM(const DegreeSequence& degrees);

template<typename T>
T pickElementUniformly(const std::vector<T>& sequence) {
    return sequence[std::uniform_int_distribution<size_t>(0, sequence.size()-1)(rng)];
}

} // namespace FastMIDyNet

#endif
