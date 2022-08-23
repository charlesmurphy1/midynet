#include "FastMIDyNet/random_graph/prior/nested_block.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

void NestedBlockPrior::_applyLabelMove(const BlockMove& move) {
    if (move.prevLabel == move.nextLabel)
        return;
    BlockIndex nestedIndex = getBlockOfIdx(move.vertexIndex, move.level-1);
    m_nestedState[move.level][nestedIndex] = move.nextLabel;

    // checking if move creates new label
    if (move.addedLabels == 1)
        createNewBlock(move);

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

    if (move.addedLabels == -1)
        destroyBlock(move);
 }


const double NestedBlockPrior::_getLogPriorRatioFromLabelMove(const BlockMove& move) const {
    std::vector<size_t> B = m_nestedBlockCountPriorPtr->getNestedState();
    double logLikelihoodBefore = m_nestedBlockCountPriorPtr->getLogLikelihoodFromNestedState(B);
    B[move.level] += move.addedLabels;
    double logLikelihoodAfter = m_nestedBlockCountPriorPtr->getLogLikelihoodFromNestedState(B);
    return logLikelihoodAfter - logLikelihoodBefore;
}

std::vector<CounterMap<BlockIndex>> NestedBlockPrior::computeNestedVertexCounts(const std::vector<BlockSequence>& nestedState) {
    std::vector<CounterMap<BlockIndex>> nestedVertexCount(nestedState.size());
    Level level = 0;
    for (const auto& b: nestedState){
        for (auto blockIdx: b)
            nestedVertexCount[level].increment(blockIdx);
        ++level;
    }
    return nestedVertexCount;
}

std::vector<CounterMap<BlockIndex>> NestedBlockPrior::computeNestedAbsoluteVertexCounts(const std::vector<BlockSequence>& nestedState) {
    std::vector<CounterMap<BlockIndex>> nestedAbsVertexCount(nestedState.size());
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

void NestedBlockPrior::createNewBlock(const BlockMove& move){
    // checking if newly created label create new level
    if (move.level == getDepth() - 1){
        m_nestedState.push_back(std::vector<BlockIndex>(move.nextLabel + 1, 0));

        m_nestedVertexCounts.push_back({});
        m_nestedVertexCounts[move.level + 1].increment(0, move.nextLabel + 1);

        m_nestedAbsVertexCounts.push_back({});
        m_nestedAbsVertexCounts[move.level + 1].increment(0, getSize());

        m_nestedBlockCountPriorPtr->createNewLevel();
    } else if (move.nextLabel < m_nestedState[move.level + 1].size()){
        m_nestedState[move.level + 1][move.nextLabel] = m_nestedState[move.level + 1][move.prevLabel];
     } else {
        m_nestedState[move.level + 1].push_back(m_nestedState[move.level + 1][move.prevLabel]);
        m_nestedVertexCounts[move.level + 1].increment(m_nestedState[move.level + 1][move.prevLabel]);
    }
}

std::vector<BlockSequence> NestedBlockPrior::reduceHierarchy(const std::vector<BlockSequence>& nestedState, Level minLevel) {
    size_t depth = nestedState.size();
    std::vector<BlockSequence> reducedState = nestedState;

    BlockIndex id, i;
    std::map<BlockIndex, BlockIndex> remap;

    for (Level l=minLevel; l<depth; ++l){
        remap.clear();
        id = 0, i = 0;
        BlockSequence relabeled;
        for (auto b : reducedState[l]){
            if (b < 0)
                continue;
            if (remap.count(b) == 0){
                remap.insert({b, id});
                if (l != depth - 1)
                    relabeled.push_back(reducedState[l + 1][b]);
                ++id;
            }
            reducedState[l][i] = remap.at(b);
            ++i;
        }
        if (l != depth - 1)
            reducedState[l+1] = relabeled;

        // remove level if each vertex is in its own community
        if (relabeled.size() == 1){
            reducedState.pop_back();
            break;
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

bool NestedBlockPrior::isValidBlockMove(const BlockMove& move) const {
    // level of move is greater than depth
    if (move.prevLabel < 0)
        return false;
        // {std::cout << "CODE 0" << std::endl; return false;}
    if (move.level >= getDepth())
        return false;
        // {std::cout << "CODE 1" << std::endl; return false;}
    // size of new partition is greater than expected blockCount
    if (m_nestedVertexCounts[move.level].size() + getAddedBlocks(move) > getNestedBlockCount(move.level) + move.addedLabels)
        return false;
        // {std::cout << "CODE 2" << std::endl; return false;}
    // if depth is 1, stop
    if (getDepth() == 1)
        return true;
    // new blockCount at level is greater than blockCount in lower level
    if (getNestedBlockCount(move.level) + move.addedLabels >= getNestedBlockCount(move.level - 1))
        return false;
        // {std::cout << "CODE 3" << std::endl; return false;}
    // if max depth is reach, stop
    if (move.level == getDepth() - 1)
        return true;
    // new blockCount at level is lesser than blockCount in upper level
    if (getNestedBlockCount(move.level) + move.addedLabels <= getNestedBlockCount(move.level + 1) and move.level < getDepth() - 2)
        return false;
        // {std::cout << "CODE 4" << std::endl; return false;}
    // block creation does not destroy block
    if (getNestedVertexCounts(move.level)[move.prevLabel] == 1 and getNestedVertexCounts(move.level)[move.nextLabel] == 0)
        return false;
        // {std::cout << "CODE 5" << std::endl; return false;}
    // destroyed block does not destroy blocks in upper layer
    if (getNestedVertexCounts(move.level + 1)[getNestedState(move.level + 1)[move.prevLabel]] == 1 and getNestedVertexCounts(move.level)[move.prevLabel] == 1)
        return false;
        // {std::cout << "CODE 6" << std::endl; return false;}
    // if creating new label, stop
    if (getNestedState(move.level + 1).size() == move.nextLabel)
        return true;
    // block of proposed label is same as block of current label
    if (getNestedState(move.level + 1)[move.prevLabel] != getNestedState(move.level + 1)[move.nextLabel])
        return false;
        // {std::cout << "CODE 7" << std::endl; return false;}
    return true;
}

const BlockSequence NestedBlockUniformPrior::sampleState(Level level) const {
    size_t N = getNestedBlockCount(level - 1);
    size_t B = getNestedBlockCount(level);
    BlockSequence blocks;
    std::uniform_int_distribution<size_t> dist(0, B - 1);
    for (size_t vertexIdx = 0; vertexIdx < N; vertexIdx++) {
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
    if (not isValidBlockMove(move))
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

    size_t N = getNestedBlockCount(level-1);
    size_t B = getNestedBlockCount(level);
    std::list<size_t> vertexCountList = sampleRandomComposition(N, B);
    std::vector<size_t> vertexCounts;
    for (auto nr : vertexCountList){
        vertexCounts.push_back(nr);
    }
    std::vector<BlockIndex> blocks;
    for(auto b : sampleRandomPermutation( vertexCounts ))
        blocks.push_back(b);
    return blocks;
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodAtLevel(Level level) const {
    int N = getNestedEffectiveBlockCount(level - 1);
    int B = getNestedEffectiveBlockCount(level);
    std::vector<size_t> nr;
    for(auto x: m_nestedVertexCounts[level])
        if (m_nestedAbsVertexCounts[level].get(x.first) > 0)
            nr.push_back(x.second);
    double logP = -logMultinomialCoefficient(m_nestedVertexCounts[level].getValues()) - logBinomialCoefficient(N - 1, B - 1);
    return logP;
}

const double NestedBlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const {
    if (not isValidBlockMove(move))
        return -INFINITY;
    if (move.prevLabel == move.nextLabel)
        return 0;

    double logLikelihoodRatio = 0;
    int addedBlocks = getAddedBlocks(move);
    int N = getNestedEffectiveBlockCount(move.level-1);
    int B = getNestedEffectiveBlockCount(move.level);
    int nrPrev = m_nestedVertexCounts[move.level][move.prevLabel];
    int nrNext = m_nestedVertexCounts[move.level][move.nextLabel];
    logLikelihoodRatio += log(nrNext + 1) - log(nrPrev);


    logLikelihoodRatio -= logBinomialCoefficient(N - 1, B + addedBlocks - 1) - logBinomialCoefficient(N - 1, B - 1);

    if (move.addedLabels != 0 and move.level != getDepth() - 1){
        BlockIndex r = getNestedBlockOfIdx(move.prevLabel, move.level + 1);
        N = getNestedEffectiveBlockCount(move.level);
        B = getNestedEffectiveBlockCount(move.level + 1);
        int nr = m_nestedVertexCounts[move.level + 1][r];
        logLikelihoodRatio += logFactorial(nr + addedBlocks) - logFactorial(nr);
        logLikelihoodRatio -= logFactorial(N + addedBlocks) - logFactorial(N);
        logLikelihoodRatio -= logBinomialCoefficient(N + addedBlocks - 1, B - 1) - logBinomialCoefficient(N - 1, B - 1);
    }

    return logLikelihoodRatio;
}



}
