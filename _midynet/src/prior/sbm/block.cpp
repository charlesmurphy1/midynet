#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/exceptions.h"

namespace FastMIDyNet{

CounterMap<size_t> BlockPrior::computeVertexCounts(const BlockSequence& state) {
    CounterMap<size_t> vertexCount;
    for (auto blockIdx: state) {
        vertexCount.increment(blockIdx);
    }
    return vertexCount;
}

void BlockPrior::checkBlockSequenceConsistencyWithVertexCounts(
    const BlockSequence& blockSeq, CounterMap<size_t> expectedVertexCounts
) {
    CounterMap<size_t> actualVertexCounts = computeVertexCounts(blockSeq);
    if (actualVertexCounts.size() != expectedVertexCounts.size())
        throw ConsistencyError("BlockPrior: size of vertex count in blockSeq is inconsistent with expected block count.");

    for (size_t i=0; i<actualVertexCounts.size(); ++i){
        auto x = actualVertexCounts[i];
        auto y = expectedVertexCounts[i];
        if (x != y){
            throw ConsistencyError("BlockPrior: actual vertex count at index "
            + std::to_string(i) + " is inconsistent with expected vertex count: "
            + std::to_string(x) + " != " + std::to_string(y) + ".");
        }
    }

}

void BlockUniformPrior::sampleState() {
    BlockSequence blockSeq(getSize());
    std::uniform_int_distribution<size_t> dist(0, getBlockCount() - 1);
    for (size_t vertexIdx = 0; vertexIdx < getSize(); vertexIdx++) {
        blockSeq[vertexIdx] = dist(rng);
    }
    setState(blockSeq);
}


const double BlockUniformPrior::getLogLikelihood() const {
    return -logMultisetCoefficient(getSize(), getBlockCount());
}

const double BlockUniformPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    size_t prevNumBlocks = m_blockCountPriorPtr->getState();
    size_t newNumBlocks = prevNumBlocks + getAddedBlocks(move);
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += -logMultisetCoefficient(getSize(), newNumBlocks);
    logLikelihoodRatio -= -logMultisetCoefficient(getSize(), prevNumBlocks);
    return logLikelihoodRatio;
}

void BlockUniformHyperPrior::sampleState() {

    std::list<size_t> vertexCountList = sampleRandomComposition(getSize(), getBlockCount());
    std::vector<size_t> vertexCounts;
    for (auto nr : vertexCountList){
        vertexCounts.push_back(nr);
    }

    BlockSequence blockSeq = sampleRandomPermutation( vertexCounts );
    setState(blockSeq);
}


const double BlockUniformHyperPrior::getLogLikelihood() const {
    return -logMultinomialCoefficient(m_vertexCounts.getValues()) - logBinomialCoefficient(getSize() - 1, getBlockCount() - 1);

}

const double BlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += logFactorial(m_vertexCounts[move.prevLabel] - 1) - logFactorial(m_vertexCounts[move.prevLabel]);
    logLikelihoodRatio += logFactorial(m_vertexCounts[move.nextLabel] + 1) - logFactorial(m_vertexCounts[move.nextLabel]);
    logLikelihoodRatio -= logBinomialCoefficient(getSize() - 1, getBlockCount() - 1 + getAddedBlocks(move)) - logBinomialCoefficient(getSize() - 1, getBlockCount() - 1);
    return logLikelihoodRatio;
}

}
