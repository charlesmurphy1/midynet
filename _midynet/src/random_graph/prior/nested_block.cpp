#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

void NestedBlockPrior::_applyLabelMove(const BlockMove& move) {
    if (move.prevLabel == move.nextLabel)
        return;
    BlockIndex nestedIndex = getBlockOfIdx(move.vertexIndex, move.level-1);
    m_nestedState[move.level][nestedIndex] = move.nextLabel;

    // checking if move creates new label
    if (move.nextLabel == getNestedBlockCountAtLevel(move.level)){
        // checking if newly created label create new level
        if (move.level == getDepth() - 1){
            m_nestedState.push_back(std::vector<BlockIndex>(move.nextLabel + 1, 0));
            m_nestedVertexCounts.push_back({});
            m_nestedVertexCounts[move.level + 1].increment(0, move.nextLabel + 1);
            m_nestedBlockCountPriorPtr->createNewLevel();
        } else {
            m_nestedState[move.level + 1].push_back(m_nestedState[move.level + 1][move.prevLabel]);
            m_nestedVertexCounts[move.level + 1].increment(m_nestedState[move.level + 1][move.prevLabel]);
        }
    }

    m_nestedBlockCountPriorPtr->setNestedStateAtLevel(m_nestedBlockCountPriorPtr->getNestedStateAtLevel(move.level) + move.addedLabels, move.level);
    m_nestedVertexCounts[move.level].decrement(move.prevLabel);
    m_nestedVertexCounts[move.level].increment(move.nextLabel);

    if (move.level == 0){
        m_vertexCounts.decrement(move.prevLabel);
        m_vertexCounts.increment(move.nextLabel);
        m_state[move.vertexIndex] = move.nextLabel;
    }
}


const double NestedBlockPrior::_getLogPriorRatioFromLabelMove(const BlockMove& move) const {
    std::vector<size_t> B = m_nestedBlockCountPriorPtr->getNestedState();
    double logLikelihoodBefore = m_nestedBlockCountPriorPtr->getLogLikelihoodFromNestedState(B);
    B[move.level] += move.addedLabels;
    double logLikelihoodAfter = m_nestedBlockCountPriorPtr->getLogLikelihoodFromNestedState(B);
    return logLikelihoodAfter - logLikelihoodBefore;
}

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

std::vector<BlockSequence> NestedBlockPrior::reduceNestedState() const {
    std::vector<BlockSequence> reducedState(getDepth());
    for (Level l=getDepth() - 1; l>=0; --l){
        std::map<BlockIndex, BlockIndex> remap;
        BlockIndex id = 0;
        for (auto b : getNestedStateAtLevel(l)){
            if (remap.count(b) == 0){
                remap.insert({b, id});
                ++id;
            }
            reducedState[l].push_back(remap[b]);
        }

    }
    return reducedState;
}
const double NestedBlockPrior::getLogLikelihood() const {
    double logLikelihood = 0;
    for (Level l=0; l<getDepth(); ++l)
        logLikelihood += getLogLikelihoodAtLevel(l);
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
    return -((double) bPrev) * log(bNext);
}

const double NestedBlockUniformPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (not isValideBlockMove(move))
        return -INFINITY;
    if (move.prevLabel == move.nextLabel)
        return 0;
    int bPrev = getNestedBlockCountAtLevel(move.level - 1);
    int bNext = getNestedBlockCountAtLevel(move.level);
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
    int bPrev = getNestedBlockCountAtLevel(level-1);
    int bNext = getNestedBlockCountAtLevel(level);

    double logP = -logMultinomialCoefficient(m_nestedVertexCounts[level].getValues()) - logBinomialCoefficient(bPrev - 1, bNext - 1);
    return logP;
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (not isValideBlockMove(move))
        return -INFINITY;
    if (move.prevLabel == move.nextLabel)
        return 0;

    double logLikelihoodRatio = 0;
    int bPrev = getNestedBlockCountAtLevel(move.level-1);
    int bNext = getNestedBlockCountAtLevel(move.level);
    logLikelihoodRatio += log(m_nestedVertexCounts[move.level][move.nextLabel] + 1) - log(m_nestedVertexCounts[move.level][move.prevLabel]);
    logLikelihoodRatio -= logFactorial(bNext + move.addedLabels) - logFactorial(bNext);
    logLikelihoodRatio -= logBinomialCoefficient(bPrev - 1, bNext + move.addedLabels - 1) - logBinomialCoefficient(bPrev - 1, bNext - 1);
    return logLikelihoodRatio;
}



}
