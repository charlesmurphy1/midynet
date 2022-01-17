#ifndef FAST_MIDYNET_UNIFORM_PROPOSER_H
#define FAST_MIDYNET_UNIFORM_PROPOSER_H


#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/sbm.h"


namespace FastMIDyNet {


class UniformBlockProposer: public BlockProposer {
    const BlockSequence* m_blocksPtr = NULL;
    const std::vector<size_t>* m_vertexCountsPtr = NULL;
    const size_t* m_blockCountPtr = NULL;
    const double m_blockCreationProbability;
    std::bernoulli_distribution m_createNewBlockDistribution;
    std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;

    public:
        UniformBlockProposer(double createNewBlockProbability=.1):
            m_createNewBlockDistribution(createNewBlockProbability),
            m_blockCreationProbability(createNewBlockProbability) {
            assertValidProbability(createNewBlockProbability);
        }
        BlockMove proposeMove(BaseGraph::VertexIndex);
        BlockMove proposeMove(){
            auto vertexIdx = m_vertexDistribution(rng);
            return proposeMove(vertexIdx);
        }
        void setUp(const StochasticBlockModelFamily& sbmGraph) ;
        double getLogProposalProbRatio(const BlockMove&) const;
        void updateProbabilities(const BlockMove&) {};
        void checkSafety() const {
            if (m_blocksPtr == nullptr)
                throw SafetyError("UniformBlockProposer: unsafe proposer since `m_blocksPtr` is NULL.");
            if (m_vertexCountsPtr == nullptr)
                throw SafetyError("UniformBlockProposer: unsafe proposer since `m_vertexCountsPtr` is NULL.");
            if (m_blockCountPtr == nullptr)
                throw SafetyError("UniformBlockProposer: unsafe proposer since `m_blockCountPtr` is NULL.");
        }

    private:
        bool creatingNewBlock(const BlockIndex& newBlock) const { return newBlock == *m_blockCountPtr; }
        bool destroyingBlock(const BlockIndex& currentBlock, const BlockIndex& newBlock) const {
            return currentBlock != newBlock && (*m_vertexCountsPtr)[currentBlock]<=1;
        }
};

} // namespace FastMIDyNet


#endif
