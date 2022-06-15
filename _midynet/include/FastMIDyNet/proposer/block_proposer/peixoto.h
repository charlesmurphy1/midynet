#ifndef FAST_MIDYNET_PEIXOTO_PROPOSER_H
#define FAST_MIDYNET_PEIXOTO_PROPOSER_H


#include "SamplableSet.hpp"

#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/random_graph.h"

namespace FastMIDyNet {


class BlockPeixotoProposer: public BlockProposer {
    const EdgeMatrix* m_edgeMatrixPtr = NULL;
    const std::vector<size_t>* m_edgeCountsPtr = NULL;
    const MultiGraph* m_graphPtr = NULL;
    const double m_blockCreationProbability;
    const double m_shift;
    mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0., 1.);
    mutable std::bernoulli_distribution m_createNewBlockDistribution;


    public:
        BlockPeixotoProposer(double createNewBlockProbability=0.1, double shift=1):
            m_createNewBlockDistribution(createNewBlockProbability),
            m_blockCreationProbability(createNewBlockProbability),
            m_shift(shift) {
            assertValidProbability(createNewBlockProbability);
        }
        const BlockMove proposeMove(const BaseGraph::VertexIndex&) const;
        void setUp(const RandomGraph& randomGraph) override;
        const double getLogProposalProb(const BlockMove& move) const;
        const double getReverseLogProposalProb(const BlockMove& move) const;
        const double getLogProposalProbRatio(const BlockMove& move) const override{
            return getReverseLogProposalProb(move) - getLogProposalProb(move);
        };
        void checkSelfSafety() const override {
            if (m_blocksPtr == nullptr)
                throw SafetyError("BlockPeixotoProposer: unsafe proposer since `m_blocksPtr` is NULL.");
            if (m_vertexCountsPtr == nullptr)
                throw SafetyError("BlockPeixotoProposer: unsafe proposer since `m_vertexCountsPtr` is NULL.");
            if (m_edgeMatrixPtr == nullptr)
                throw SafetyError("BlockPeixotoProposer: unsafe proposer since `m_edgeMatrixPtr` is NULL.");
            if (m_edgeCountsPtr == nullptr)
                throw SafetyError("BlockPeixotoProposer: unsafe proposer since `m_edgeCountsPtr` is NULL.");
            if (m_graphPtr == nullptr)
                throw SafetyError("BlockPeixotoProposer: unsafe proposer since `m_graphPtr` is NULL.");
        }
    protected:
        IntMap<std::pair<BlockIndex, BlockIndex>> getEdgeMatrixDiff(const BlockMove& move) const ;
        IntMap<BlockIndex> getEdgeCountsDiff(const BlockMove& move) const ;
};

} // namespace FastMIDyNet


#endif
