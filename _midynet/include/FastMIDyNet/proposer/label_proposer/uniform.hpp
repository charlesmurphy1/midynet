#ifndef FAST_MIDYNET_UNIFORM_PROPOSER_H
#define FAST_MIDYNET_UNIFORM_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label_proposer/label_proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"


namespace FastMIDyNet {

template<typename Label>
class LabelUniformProposer: public LabelProposer<Label> {
    const double m_labelCreationProbability;
    mutable std::bernoulli_distribution m_createNewLabelDistribution;

public:
    LabelUniformProposer(double createNewLabelProbrobability=.1):
        m_createNewLabelDistribution(createNewLabelProbrobability),
        m_labelCreationProbability(createNewLabelProbrobability) {
        assertValidProbability(createNewLabelProbrobability);
    }
    const LabelMove<Label> proposeMove(const BaseGraph::VertexIndex&) const;
    const double getLogProposalProbRatio(const LabelMove<Label>&) const override;
};

class BlockUniformProposer: public LabelUniformProposer<BlockIndex>{
public:
    using LabelUniformProposer<BlockIndex>::LabelUniformProposer;
};

template<typename Label>
const LabelMove<Label> LabelUniformProposer<Label>::proposeMove(const BaseGraph::VertexIndex& movedVertex) const {
    size_t B = this->m_labelCountsPtr->size();
    const auto& labels = *this->m_labelsPtr;
    if (B == 1 && m_labelCreationProbability == 0)
        return {movedVertex, labels[movedVertex], labels[movedVertex]};


    const BlockIndex& currentBlock = labels[movedVertex];

    BlockIndex newBlock;
    if (m_createNewLabelDistribution(rng)){
        newBlock = B;
    }
    else if (B > 1) {
        newBlock = std::uniform_int_distribution<BlockIndex>(0, B - 1)(rng);
    } else {
        return {0, labels[0], labels[0]};
    }
    LabelMove<Label> move = {movedVertex, currentBlock, newBlock};
    return {movedVertex, currentBlock, newBlock};
}

template<typename Label>
const double LabelUniformProposer<Label>::getLogProposalProbRatio(const LabelMove<Label>& move) const {
    size_t B = this->m_labelCountsPtr->size();
    if (this->creatingNewLabel(move))
        return -log(m_labelCreationProbability) + log(1-m_labelCreationProbability) - log(B);
    else if (this->destroyingLabel(move))
        return log(B-1) - log(1-m_labelCreationProbability) + log(m_labelCreationProbability);
    return 0;
}

} // namespace FastMIDyNet


#endif
