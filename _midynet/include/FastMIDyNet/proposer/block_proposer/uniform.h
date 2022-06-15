#ifndef FAST_MIDYNET_UNIFORM_PROPOSER_H
#define FAST_MIDYNET_UNIFORM_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"


namespace FastMIDyNet {


class BlockUniformProposer: public BlockProposer {
    const double m_blockCreationProbability;
    mutable std::bernoulli_distribution m_createNewBlockDistribution;
    mutable std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;

public:
    BlockUniformProposer(double createNewBlockProbability=.1):
        m_createNewBlockDistribution(createNewBlockProbability),
        m_blockCreationProbability(createNewBlockProbability) {
        assertValidProbability(createNewBlockProbability);
    }
    const BlockMove proposeRawMove(BaseGraph::VertexIndex) const;
    const BlockMove proposeRawMove() const override{
        auto vertexIdx = m_vertexDistribution(rng);
        return proposeRawMove(vertexIdx);
    }
    void setUp(const RandomGraph& randomGraph) override;
    const double getLogProposalProbRatio(const BlockMove&) const override;
    void checkSelfSafety() const override{
        if (m_blocksPtr == nullptr)
            throw SafetyError("BlockUniformProposer: unsafe proposer since `m_blocksPtr` is NULL.");
        if (m_vertexCountsPtr == nullptr)
            throw SafetyError("BlockUniformProposer: unsafe proposer since `m_vertexCountsPtr` is NULL.");
    }
};

} // namespace FastMIDyNet


#endif
