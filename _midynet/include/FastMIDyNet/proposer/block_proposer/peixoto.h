#ifndef FAST_MIDYNET_PEIXOTO_PROPOSER_H
#define FAST_MIDYNET_PEIXOTO_PROPOSER_H


#include "SamplableSet.hpp"

#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/block_proposer/block_proposer.h"
#include "FastMIDyNet/random_graph/sbm.h"


namespace FastMIDyNet {


class PeixotoBlockProposer: public BlockProposer {
    const BlockSequence* m_blocksPtr = NULL;
    const std::vector<size_t>* m_vertexCountsPtr = NULL;
    const size_t* m_blockCountPtr = NULL;
    const EdgeMatrix* m_edgeMatrixPtr = NULL;
    const std::vector<size_t>* m_edgeCountsPtr = NULL;
    const MultiGraph* m_graphPtr = NULL;
    const double m_blockCreationProbability;
    const double m_shift;
    std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0., 1.);
    std::bernoulli_distribution m_createNewBlockDistribution;
    std::uniform_int_distribution<BaseGraph::VertexIndex> m_vertexDistribution;


    public:
        PeixotoBlockProposer(double createNewBlockProbability=0.1, double shift=1):
            m_createNewBlockDistribution(createNewBlockProbability),
            m_blockCreationProbability(createNewBlockProbability),
            m_shift(shift) {
            assertValidProbability(createNewBlockProbability);
        }
        BlockMove proposeMove(BaseGraph::VertexIndex);
        BlockMove proposeMove(){
            auto vertexIdx = m_vertexDistribution(rng);
            return proposeMove(vertexIdx);
        }
        void setUp(const StochasticBlockModelFamily& sbmGraph);
        double getLogProposalProb(const BlockMove& move) const;
        double getReverseLogProposalProb(const BlockMove& move) const;
        double getLogProposalProbRatio(const BlockMove& move) const{
            return getReverseLogProposalProb(move) - getLogProposalProb(move);
        };
        void updateProbabilities(const BlockMove&) {};
        void checkSafety() const {
            if (m_blocksPtr == nullptr)
                throw SafetyError("PeixotoBlockProposer: unsafe proposer since `m_blocksPtr` is NULL.");
            if (m_vertexCountsPtr == nullptr)
                throw SafetyError("PeixotoBlockProposer: unsafe proposer since `m_vertexCountsPtr` is NULL.");
            if (m_blockCountPtr == nullptr)
                throw SafetyError("PeixotoBlockProposer: unsafe proposer since `m_blockCountPtr` is NULL.");
            if (m_edgeMatrixPtr == nullptr)
                throw SafetyError("PeixotoBlockProposer: unsafe proposer since `m_edgeMatrixPtr` is NULL.");
            if (m_edgeCountsPtr == nullptr)
                throw SafetyError("PeixotoBlockProposer: unsafe proposer since `m_edgeCountsPtr` is NULL.");
            if (m_graphPtr == nullptr)
                throw SafetyError("PeixotoBlockProposer: unsafe proposer since `m_graphPtr` is NULL.");
        }
    private:
        IntMap<std::pair<BlockIndex, BlockIndex>> getEdgeMatrixDiff(const BlockMove& move) const ;
        IntMap<BlockIndex> getEdgeCountsDiff(const BlockMove& move) const ;
        bool creatingNewBlock(const BlockIndex& newBlock) const { return newBlock == *m_blockCountPtr; }
        bool destroyingBlock(const BlockIndex& currentBlock, const BlockIndex& newBlock) const {
            return currentBlock != newBlock && (*m_vertexCountsPtr)[currentBlock]<=1;
        }
};

} // namespace FastMIDyNet


#endif
