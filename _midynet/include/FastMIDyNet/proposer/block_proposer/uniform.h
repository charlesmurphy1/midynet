#ifndef FAST_MIDYNET_UNIFORM_PROPOSER_H
#define FAST_MIDYNET_UNIFORM_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {


class BlockUniformProposer: public BlockProposer {
    const BlockSequence* m_blocksPtr = nullptr;
    const std::vector<size_t>* m_vertexCountsPtr = nullptr;
    const size_t* m_blockCountPtr = nullptr;
    const double m_blockCreationProbability;
    mutable std::bernoulli_distribution m_createNewBlockDistribution;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;

    public:
        BlockUniformProposer(double createNewBlockProbability=.1):
            m_createNewBlockDistribution(createNewBlockProbability),
            m_blockCreationProbability(createNewBlockProbability) {
            assertValidProbability(createNewBlockProbability);
        }
        BlockMove proposeMove(BaseGraph::VertexIndex) const;
        BlockMove proposeMove() const override{
            auto vertexIdx = m_vertexDistribution(rng);
            return proposeMove(vertexIdx);
        }
        void setUp(const RandomGraph& randomGraph) override;
        const double getLogProposalProbRatio(const BlockMove&) const override;
        void checkSelfSafety() const override{
            if (m_blocksPtr == nullptr)
                throw SafetyError("BlockUniformProposer: unsafe proposer since `m_blocksPtr` is NULL.");
            if (m_vertexCountsPtr == nullptr)
                throw SafetyError("BlockUniformProposer: unsafe proposer since `m_vertexCountsPtr` is NULL.");
            if (m_blockCountPtr == nullptr)
                throw SafetyError("BlockUniformProposer: unsafe proposer since `m_blockCountPtr` is NULL.");
        }

    private:
        bool creatingNewBlock(const BlockIndex& newBlock) const { return newBlock == *m_blockCountPtr; }
        bool destroyingBlock(const BlockIndex& currentBlock, const BlockIndex& newBlock) const {
            return currentBlock != newBlock && (*m_vertexCountsPtr)[currentBlock]<=1;
        }
};

} // namespace FastMIDyNet


#endif
