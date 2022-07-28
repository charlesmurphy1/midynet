#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "FastMIDyNet/random_graph/prior/block.h"
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
    std::string prefix, const BlockSequence& blockSeq, CounterMap<size_t> expectedVertexCounts
) {
    CounterMap<size_t> actualVertexCounts = computeVertexCounts(blockSeq);
    if (actualVertexCounts.size() != expectedVertexCounts.size())
        throw ConsistencyError(
            prefix,
            "blockSeq", "B=" + std::to_string(actualVertexCounts.size()),
            "vertexCounts", "B=" + std::to_string(expectedVertexCounts.size())
        );

    for (size_t i=0; i<actualVertexCounts.size(); ++i){
        auto x = actualVertexCounts[i];
        auto y = expectedVertexCounts[i];
        if (x != y){
            throw ConsistencyError(
                prefix,
                "blockSeq", "counts=" + std::to_string(actualVertexCounts.size()),
                "vertexCounts", "value=" + std::to_string(expectedVertexCounts.size()),
                "r=" + std::to_string(i)
            );
        }
    }

}

void BlockUniformPrior::sampleState() {
    BlockSequence blockSeq(getSize());
    std::uniform_int_distribution<size_t> dist(0, getBlockCount() - 1);
    for (size_t vertexIdx = 0; vertexIdx < getSize(); vertexIdx++) {
        blockSeq[vertexIdx] = dist(rng);
    }
    m_state = blockSeq;
    m_vertexCounts = computeVertexCounts(m_state);
}


const double BlockUniformPrior::getLogLikelihood() const {
    return -(getSize() * log(getBlockCount()));
}

const double BlockUniformPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (m_vertexCounts.size() + getAddedBlocks(move) > m_blockCountPriorPtr->getState() + move.addedLabels)
        return -INFINITY;
    size_t prevNumBlocks = m_blockCountPriorPtr->getState();
    size_t newNumBlocks = prevNumBlocks + move.addedLabels;
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += -(double)getSize() * log(newNumBlocks);
    logLikelihoodRatio -= -(double)getSize() * log(prevNumBlocks);
    return logLikelihoodRatio;
}

void BlockUniformHyperPrior::sampleState() {

    std::list<size_t> vertexCountList = sampleRandomComposition(getSize(), getBlockCount());
    std::vector<size_t> vertexCounts;
    for (auto nr : vertexCountList){
        vertexCounts.push_back(nr);
    }

    m_state = sampleRandomPermutation( vertexCounts );
    m_vertexCounts = computeVertexCounts(m_state);
}


const double BlockUniformHyperPrior::getLogLikelihood() const {
    return -logMultinomialCoefficient(m_vertexCounts.getValues()) - logBinomialCoefficient(getSize() - 1, getBlockCount() - 1);

}

const double BlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (m_vertexCounts.size() + getAddedBlocks(move) != getBlockCount() + move.addedLabels)
        return -INFINITY;
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += logFactorial(m_vertexCounts[move.prevLabel] - 1) - logFactorial(m_vertexCounts[move.prevLabel]);
    logLikelihoodRatio += logFactorial(m_vertexCounts[move.nextLabel] + 1) - logFactorial(m_vertexCounts[move.nextLabel]);
    logLikelihoodRatio -= logBinomialCoefficient(getSize() - 1, getBlockCount() + move.addedLabels - 1) - logBinomialCoefficient(getSize() - 1, getBlockCount() - 1);
    return logLikelihoodRatio;
}

}
