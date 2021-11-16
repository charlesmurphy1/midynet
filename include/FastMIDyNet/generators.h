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
std::list<int> sampleUniformlySequenceWithoutReplacement(size_t n, size_t k, RNG& rng);
BaseGraph::UndirectedMultigraph generateDCSBM(const std::vector<size_t>& vertexBlocks,
        const Matrix<size_t>& blockEdgeMatrix, const std::vector<size_t>& degrees, RNG& rng);
BaseGraph::UndirectedMultigraph generateSBM(const std::vector<size_t>& vertexBlocks,
        const Matrix<size_t>& blockEdgeMatrix, RNG& rng);
FastMIDyNet::MultiGraph generateCM(const std::vector<size_t>& degrees);

template<typename T>
T pickElementUniformly(const std::vector<T>& sequence, RNG& rng) {
    return sequence[std::uniform_int_distribution<size_t>(0, sequence.size()-1)(rng)];
}

} // namespace FastMIDyNet

#endif
