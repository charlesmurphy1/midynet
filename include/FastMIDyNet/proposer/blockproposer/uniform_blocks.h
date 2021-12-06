#ifndef FAST_MIDYNET_UNIFORM_BLOCKPROPOSER_H
#define FAST_MIDYNET_UNIFORM_BLOCKPROPOSER_H


#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/blockproposer/blockproposer.h"
#include "FastMIDyNet/random_graph/sbm.h"


namespace FastMIDyNet {


class UniformBlockProposer: public BlockProposer {
    const BlockSequence* m_blockSequencePtr = NULL;
    size_t m_blockCount = 0;
    const double m_blockCreationProbability;
    std::bernoulli_distribution m_createNewBlockDistribution;
    std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;
    std::vector<size_t> m_vertexCountInBlocks;

    public:
        UniformBlockProposer(size_t graphSize, double createNewBlockProbability=.1);
        BlockMove proposeMove();
        void setup(const StochasticBlockModelFamily& sbmGraph);
        double getLogProposalProbRatio(const BlockMove&) const;
        void updateProbabilities(const BlockMove&);

        const std::vector<size_t> getVertexCountInBlocks() { return m_vertexCountInBlocks; }
        const size_t getInternalBlockCount() { return m_blockCount; }
        void checkConsistency();
    private:
        bool creatingNewBlock(const BlockIndex& newBlock) const { return newBlock == m_blockCount; }
        bool destroyingBlock(const BlockIndex& currentBlock, const BlockIndex& newBlock) const {
            return currentBlock != newBlock && m_vertexCountInBlocks[currentBlock]<=1;
        }
};

} // namespace FastMIDyNet


#endif
