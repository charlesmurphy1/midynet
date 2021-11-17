#ifndef FAST_MIDYNET_GENERATORS
#define FAST_MIDYNET_GENERATORS

#include <random>
#include <vector>
#include <list>
#include "BaseGraph/undirected_multigraph.h"
#include "BaseGraph/types.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


int generateCategorical(const std::vector<double>& probs, RNG& rng);

std::vector<size_t> sampleUniformlySequenceWithoutReplacement(size_t n, size_t k, RNG& rng);
std::list<size_t> sampleRandomComposition(size_t n, size_t k, RNG& rng);
std::list<size_t> sampleRandomWeakComposition(size_t n, size_t k, RNG& rng);

double logMultinomialCoefficient(std::list<size_t> sequence);
std::list<size_t> sampleRandomRestrictedPartition(size_t n, size_t k, RNG& rng, size_t numberOfSteps=0);

BaseGraph::UndirectedMultigraph generateDCSBM(const BlockSequence& vertexBlocks,
        const EdgeMatrix& blockEdgeMatrix, const DegreeSequence& degrees, RNG& rng);
BaseGraph::UndirectedMultigraph generateSBM(const BlockSequence& vertexBlocks,
        const EdgeMatrix& blockEdgeMatrix, RNG& rng);
FastMIDyNet::MultiGraph generateCM(const DegreeSequence& degrees);

template<typename T>
T pickElementUniformly(const std::vector<T>& sequence, RNG& rng) {
    return sequence[std::uniform_int_distribution<size_t>(0, sequence.size()-1)(rng)];
}

} // namespace FastMIDyNet

#endif
