#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

void NestedBlockPrior::_applyLabelMove(const BlockMove& move) {
    if (move.prevLabel == move.nextLabel or not isValideBlockMove(move))
        return;
    BlockIndex nestedIndex = getBlockOfIdx(move.vertexIndex, move.level-1);
    m_nestedState[move.level][nestedIndex] = move.nextLabel;

    // checking if move creates new label
    if (move.nextLabel == getNestedBlockCount(move.level) and move.addedLabels == 1){
        // checking if newly created label create new level
        if (move.level == getDepth() - 1){
            m_nestedState.push_back(std::vector<BlockIndex>(move.nextLabel + 1, 0));

            m_nestedVertexCounts.push_back({});
            m_nestedVertexCounts[move.level + 1].increment(0, move.nextLabel + 1);

            m_nestedAbsVertexCounts.push_back({});
            m_nestedAbsVertexCounts[move.level + 1].increment(0, getSize());

            m_nestedBlockCountPriorPtr->createNewLevel();
        } else {
            m_nestedState[move.level + 1].push_back(m_nestedState[move.level + 1][move.prevLabel]);
            m_nestedVertexCounts[move.level + 1].increment(m_nestedState[move.level + 1][move.prevLabel]);
        }
    }
    // // checking if move destroys label
    // else if (getAddedBlocks(move) == -1){
    //     m_nestedState[move.level] = reducePartition(m_nestedState[move.level]);
    //     if (move.level == getDepth() - 2){
    //         m_nestedState[move.level + 1].erase(m_nestedState[move.level + 1].begin() + move.prevLabel);
    //         if (m_nestedState[move.level + 1].size() == 1){
    //             m_nestedState.pop_back();
    //             m_nestedVertexCounts.pop_back();
    //         }
    //     } else {
    //         size_t newBlockCount = getMaxBlockCountFromPartition(m_nestedState[move.level]);
    //
    //     }
    //
    // }



    m_nestedBlockCountPriorPtr->setNestedState(m_nestedBlockCountPriorPtr->getNestedState(move.level) + move.addedLabels, move.level);

    m_nestedVertexCounts[move.level].decrement(move.prevLabel);
    m_nestedVertexCounts[move.level].increment(move.nextLabel);

    size_t nr = (move.level==0) ? 1 : m_nestedAbsVertexCounts[move.level-1][getBlockOfIdx(move.vertexIndex, move.level-1)];
    m_nestedAbsVertexCounts[move.level].decrement(move.prevLabel, nr);
    m_nestedAbsVertexCounts[move.level].increment(move.nextLabel, nr);

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
    std::vector<CounterMap<size_t>> nestedVertexCount(nestedState.size());
    Level level = 0;
    for (const auto& b: nestedState){
        for (auto blockIdx: b)
            nestedVertexCount[level].increment(blockIdx);
        ++level;
    }
    return nestedVertexCount;
}

std::vector<CounterMap<size_t>> NestedBlockPrior::computeNestedAbsoluteVertexCounts(const std::vector<BlockSequence>& nestedState) {
    std::vector<CounterMap<size_t>> nestedAbsVertexCount(nestedState.size());
    Level level = 0;
    size_t nr, id;
    for (const auto& b: nestedState){
        id = 0;
        for (auto blockIdx: b) {
            nr = (level == 0) ? 1 : nestedAbsVertexCount[level-1][id];
            nestedAbsVertexCount[level].increment(blockIdx, nr);
            ++id;
        }
        ++level;
    }
    return nestedAbsVertexCount;
}

std::vector<BlockSequence> NestedBlockPrior::reduceHierarchy(const std::vector<BlockSequence>& nestedState) {
    size_t depth = nestedState.size();
    std::vector<BlockSequence> reducedState;
    std::vector<CounterMap<BlockIndex>> vertexCounts = computeNestedAbsoluteVertexCounts(nestedState);

    BlockIndex id = 0, i;
    std::map<BlockIndex, BlockIndex> remap;

    for (Level l=0; l<depth; ++l){
        remap.clear();
        id = 0;
        i = 0;
        reducedState.push_back({});
        for (auto b : nestedState[l]){
            if (l != 0 and vertexCounts[l-1][i++] == 0)
                continue;
            if (remap.count(b) == 0){
                remap.insert({b, id});
                ++id;
            }
            reducedState.back().push_back(remap.at(b));
        }
        // remove level if each vertex is in its own community
        if (reducedState.back().size() == 1){
            reducedState.pop_back();
            break;
        } else if (l!=0 and reducedState.back().size() == vertexCounts[l].size())
            reducedState.pop_back();
    }



    return reducedState;
}

const double NestedBlockPrior::getLogLikelihood() const {
    double logLikelihood = 0;
    for (Level l=0; l<getDepth(); ++l)
        logLikelihood += getLogLikelihoodAtLevel(l);
    return logLikelihood;
}

bool NestedBlockPrior::isValideBlockMove(const BlockMove& move) const {
    // level of move is greater than depth
    if (move.level >= getDepth())
        return false;
    // size of new partition is greater than expected blockCount
    if (m_nestedVertexCounts[move.level].size() + getAddedBlocks(move) > getNestedBlockCount(move.level) + move.addedLabels)
        return false;
    // if depth is 1, stop
    if (getDepth() == 1)
        return true;
    // new blockCount at level is greater than blockCount in lower level
    if (getNestedBlockCount(move.level) + move.addedLabels >= getNestedBlockCount(move.level - 1))
        return false;
    // new blockCount at level is lesser than blockCount in upper level
    if (getNestedBlockCount(move.level) + move.addedLabels <= getNestedBlockCount(move.level + 1))
        return false;
    // if max depth is reach, stop
    if (move.level == getDepth() - 1)
        return true;
    // new blockCount at level is lesser than blockCount in upper level
    if (getNestedVertexCounts(move.level + 1)[getNestedState(move.level + 1)[move.prevLabel]] == 1 and getAddedBlocks(move) == -1)
        return false;
    // if creating new label, stop
    if (getNestedState(move.level + 1).size() == move.nextLabel)
        return true;
    // block of proposed label is same as block of current label
    if (getNestedState(move.level + 1)[move.prevLabel] != getNestedState(move.level + 1)[move.nextLabel])
        return false;
    return true;
}


const BlockSequence NestedBlockUniformPrior::sampleState(Level level) const {
    size_t bPrev = getNestedBlockCount(level - 1);
    size_t bNext = getNestedBlockCount(level);
    BlockSequence blocks;
    std::uniform_int_distribution<size_t> dist(0, bNext - 1);
    for (size_t vertexIdx = 0; vertexIdx < bPrev; vertexIdx++) {
        blocks.push_back(dist(rng));
    }
    return blocks;
}


const double NestedBlockUniformPrior::getLogLikelihoodAtLevel(Level level) const {
    size_t bPrev = (level == 0) ? getSize() : getNestedBlockCount()[level - 1];
    size_t bNext = getNestedBlockCount(level);
    return -((double) bPrev) * log(bNext);
}

const double NestedBlockUniformPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (not isValideBlockMove(move))
        return -INFINITY;
    if (move.prevLabel == move.nextLabel)
        return 0;
    int bPrev = getNestedBlockCount(move.level - 1);
    int bNext = getNestedBlockCount(move.level);
    double logLikelihoodRatio = 0;
    logLikelihoodRatio += -bPrev * log(bNext + move.addedLabels);
    logLikelihoodRatio -= -bPrev * log(bNext);
    return logLikelihoodRatio;
}

const BlockSequence NestedBlockUniformHyperPrior::sampleState(Level level) const {

    size_t bPrev, bNext;
    bPrev = (level == 0) ? getSize() : getNestedBlockCount()[level - 1];
    bNext = getNestedBlockCount(level);
    std::list<size_t> vertexCountList = sampleRandomComposition(bPrev, bNext);
    std::vector<size_t> vertexCounts;
    for (auto nr : vertexCountList){
        vertexCounts.push_back(nr);
    }
    return sampleRandomPermutation( vertexCounts );
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodAtLevel(Level level) const {
    int bPrev = (level==0) ? getSize() : m_nestedAbsVertexCounts[level - 1].size();
    int bNext = m_nestedAbsVertexCounts[level].size();
    std::vector<size_t> nr;
    for(auto x: m_nestedVertexCounts[level])
        if (m_nestedAbsVertexCounts[level].get(x.first) > 0)
            nr.push_back(x.second);

    double logP = -logMultinomialCoefficient(m_nestedVertexCounts[level].getValues()) - logBinomialCoefficient(bPrev - 1, bNext - 1);
    return logP;
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (not isValideBlockMove(move))
        return -INFINITY;
    if (move.prevLabel == move.nextLabel)
        return 0;

    double logLikelihoodRatio = 0;
    int bPrev = getNestedBlockCount(move.level-1);
    int bNext = getNestedBlockCount(move.level);
    logLikelihoodRatio += log(m_nestedVertexCounts[move.level][move.nextLabel] + 1) - log(m_nestedVertexCounts[move.level][move.prevLabel]);
    logLikelihoodRatio -= logFactorial(bNext + move.addedLabels) - logFactorial(bNext);
    logLikelihoodRatio -= logBinomialCoefficient(bPrev - 1, bNext + move.addedLabels - 1) - logBinomialCoefficient(bPrev - 1, bNext - 1);
    return logLikelihoodRatio;
}



}
