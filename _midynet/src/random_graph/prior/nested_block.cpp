#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{


std::vector<CounterMap<size_t>> NestedBlockPrior::computeNestedVertexCounts(const std::vector<BlockSequence>& nestedState) {
    std::vector<CounterMap<size_t>> nestedVertexCount;
    Level level = 0;
    for (const auto& b: nestedState){
        nestedVertexCount.push_back({});
        for (auto blockIdx: b) {
            nestedVertexCount[level].increment(blockIdx);
        }
        ++level;
    }
    return nestedVertexCount;
}

const double NestedBlockPrior::getLogLikelihood() const {
    double logLikelihood = 0;
    for (Level l=0; l<getDepth(); ++l)
        logLikelihood -= getLogLikelihoodAtLevel(l);
    return logLikelihood;
}


const BlockSequence NestedBlockUniformPrior::sampleStateAtLevel(Level level) const {
    size_t bPrev = getNestedBlockCountAtLevel(level - 1);
    size_t bNext = getNestedBlockCountAtLevel(level);
    BlockSequence blocks;
    std::uniform_int_distribution<size_t> dist(0, bNext - 1);
    for (size_t vertexIdx = 0; vertexIdx < bPrev; vertexIdx++) {
        blocks.push_back(dist(rng));
    }
    return blocks;
}


const double NestedBlockUniformPrior::getLogLikelihoodAtLevel(Level level) const {
    size_t bPrev = (level == 0) ? getSize() : getNestedBlockCount()[level - 1];
    size_t bNext = getNestedBlockCount()[level];
    return -bPrev * log(bNext);
}

const double NestedBlockUniformPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (m_nestedVertexCounts[move.level].size() + getAddedBlocks(move) > m_blockCountPriorPtr->getState() + move.addedLabels)
        return -INFINITY;
    size_t bPrev = (move.level == 0) ? getSize() : getNestedBlockCount()[move.level - 1];
    size_t bNext = getNestedBlockCount()[move.level];
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += -bPrev * log(bNext + move.addedLabels);
    logLikelihoodRatio -= -bPrev * log(bNext);
    return logLikelihoodRatio;
}

const BlockSequence NestedBlockUniformHyperPrior::sampleStateAtLevel(Level level) const {

    size_t bPrev, bNext;
    bPrev = (level == 0) ? getSize() : getNestedBlockCount()[level - 1];
    bNext = getNestedBlockCount()[level];
    std::list<size_t> vertexCountList = sampleRandomComposition(bPrev, bNext);
    std::vector<size_t> vertexCounts;
    for (auto nr : vertexCountList){
        vertexCounts.push_back(nr);
    }
    return sampleRandomPermutation( vertexCounts );
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodAtLevel(Level level) const {
    size_t bPrev = (level == 0) ? getSize() : getNestedBlockCount()[level-1];
    size_t bNext = getNestedBlockCount()[level];
    return -logMultinomialCoefficient(m_nestedVertexCounts[level].getValues()) - logBinomialCoefficient(bPrev - 1, bNext - 1);
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (m_nestedVertexCounts[move.level].size() + getAddedBlocks(move) != getNestedBlockCount()[move.level] + move.addedLabels)
        return -INFINITY;
    double logLikelihoodRatio = 0;
    size_t bPrev = (move.level == 0) ? getSize() : getNestedBlockCount()[move.level-1];
    size_t bNext = getNestedBlockCount()[move.level];
    logLikelihoodRatio += logFactorial(m_nestedVertexCounts[move.level][move.prevLabel] - 1) - logFactorial(m_nestedVertexCounts[move.level][move.prevLabel]);
    logLikelihoodRatio += logFactorial(m_nestedVertexCounts[move.level][move.nextLabel] + 1) - logFactorial(m_nestedVertexCounts[move.level][move.nextLabel]);
    logLikelihoodRatio -= logBinomialCoefficient(bPrev - 1, bNext + move.addedLabels - 1) - logBinomialCoefficient(bPrev - 1, bNext - 1);
    return logLikelihoodRatio;
}



}
