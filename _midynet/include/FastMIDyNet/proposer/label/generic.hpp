#ifndef FAST_MIDYNET_GENERIC_PROPOSER_H
#define FAST_MIDYNET_GENERIC_PROPOSER_H


#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/proposer/label/label_proposer.hpp"
#include "FastMIDyNet/random_graph/random_graph.hpp"


namespace FastMIDyNet {

// template<typename Label>
// class LabelGenericProposer: public LabelProposer<Label> {
// // protected:
// //     bool creatingNewBlock(const BlockMove&) const { return false; }
// //     bool destroyingBlock(const BlockMove&) const { return false; }
// public:
//     using LabelProposer<Label>::LabelProposer;
//     const LabelMove<Label> proposeMove(const BaseGraph::VertexIndex& vertex) const override{
//         return {vertex, (*LabelProposer<Label>::m_labelsPtr)[vertex], (*LabelProposer<Label>::m_labelsPtr)[vertex]};
//     }
//     const double getLogProposalProbRatio(const LabelMove<Label>&) const override { return 0;};
//     void checkSelfSafety() const override { }
// };

} // namespace FastMIDyNet


#endif
