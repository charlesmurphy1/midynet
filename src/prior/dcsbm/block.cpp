#include <algorithm>
#include <random>
#include <vector>

#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"


using namespace std;


namespace FastMIDyNet{
    vector<size_t> BlockPrior::getVertexCount(const BlockSequence& state) const {
        size_t numBlocks = *max_element(state.begin(), state.end()) + 1;
        vector<size_t> vertexCount(numBlocks);

        for (auto blockIdx: state) {
            vertexCount[blockIdx] ++;
        }

        return vertexCount;
    };

    BlockSequence BlockUniformPrior::sample() {
        BlockSequence blockSeq(getSize());
        uniform_int_distribution<size_t> dist(0, getBlockCount() - 1);
        for (size_t vertexIdx = 0; vertexIdx < getSize(); vertexIdx++) {
            blockSeq[vertexIdx] = dist(rng);
        }
        return blockSeq;
    };

    double BlockUniformPrior::getLogLikelihood(const BlockSequence& state) const {
        return -logMultisetCoefficient(getSize(), getBlockCount());
    };


    double BlockUniformPrior::getLogLikelihoodRatio(const BlockMove& move) const {
        size_t prevNumBlocks = m_blockCountPrior.getState();
        size_t newNumBlocks = m_blockCountPrior.getStateAfterMove(move);
        double logLikelihoodRatio = 0;
        logLikelihoodRatio += -logMultisetCoefficient(getSize(), newNumBlocks);
        logLikelihoodRatio -= -logMultisetCoefficient(getSize(), prevNumBlocks);
        return logLikelihoodRatio;


    };

    void BlockUniformPrior::applyMove(const BlockMove& move) {
        if (!m_isProcessed)
            m_state[move.vertexIdx] = move.nextBlockIdx;
        m_isProcessed = true;
    };

    void BlockUniformPrior::checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) {
        size_t actualBlockCount = *max_element(blockSeq.begin(), blockSeq.end()) + 1;
        if (actualBlockCount != expectedBlockCount)
            throw ConsistencyError("BlockUniformPrior: blockSeq is inconsistent with expected block count.");

    };

}
