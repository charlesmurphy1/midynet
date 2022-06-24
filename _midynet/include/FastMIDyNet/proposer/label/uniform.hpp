#ifndef FAST_MIDYNET_UNIFORM_PROPOSER_H
#define FAST_MIDYNET_UNIFORM_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"


namespace FastMIDyNet {

template<typename Label>
class LabelUniformProposer: public LabelProposer<Label> {

public:
    LabelUniformProposer(double labelCreationProb=.1):
        LabelProposer<Label>(labelCreationProb){ }
    const LabelMove<Label> proposeMove(const BaseGraph::VertexIndex&) const;
    const double getLogProposalProbRatio(const LabelMove<Label>&) const override;
};

class BlockUniformProposer: public LabelUniformProposer<BlockIndex>{
public:
    using LabelUniformProposer<BlockIndex>::LabelUniformProposer;
};

template<typename Label>
const LabelMove<Label> LabelUniformProposer<Label>::proposeMove(const BaseGraph::VertexIndex& movedVertex) const {
    size_t B = LabelProposer<Label>::m_labelCountsPtr->size();
    const auto& labels = *LabelProposer<Label>::m_labelsPtr;
    if (B == 1 &&  LabelProposer<Label>::m_labelCreationProb == 0)
        return {movedVertex, labels[movedVertex], labels[movedVertex]};


    const BlockIndex& currentBlock = labels[movedVertex];

    BlockIndex newBlock;
    if (LabelProposer<Label>::m_createNewLabelDistribution(rng)){
        newBlock = B;
    }
    else if (B > 1) {
        newBlock = std::uniform_int_distribution<BlockIndex>(0, B - 1)(rng);
    } else {
        return {0, labels[0], labels[0]};
    }
    return {movedVertex, currentBlock, newBlock};
}

template<typename Label>
const double LabelUniformProposer<Label>::getLogProposalProbRatio(const LabelMove<Label>& move) const {
    size_t B = this->m_labelCountsPtr->size();
    if (this->creatingNewLabel(move))
        return -log( LabelProposer<Label>::m_labelCreationProb) + log(1- LabelProposer<Label>::m_labelCreationProb) - log(B);
    else if (this->destroyingLabel(move))
        return log(B-1) - log(1- LabelProposer<Label>::m_labelCreationProb) + log( LabelProposer<Label>::m_labelCreationProb);
    return 0;
}

} // namespace FastMIDyNet


#endif
