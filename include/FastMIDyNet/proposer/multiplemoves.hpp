#ifndef FAST_MIDYNET_MULTIPLEMOVE_PROPOSER_H
#define FAST_MIDYNET_MULTIPLEMOVE_PROPOSER_H

#include "proposer.hpp"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/utility.h"


namespace FastMIDyNet {


template<typename MoveType>
class MultipleMovesProposer: public Proposer<MoveType> {
    std::vector<Proposer<MoveType>*>& m_proposers;
    std::vector<double> m_moveWeights;
    unsigned int m_proposedMoveType=0;
    std::discrete_distribution<unsigned int> m_moveTypeDistribution;

    public:
        MultipleMovesProposer(std::vector<Proposer<MoveType>*>& proposers, std::vector<double> moveWeights);
        MoveType operator()();
        double getProposalProb(const MoveType&) const;
        void updateProbabilities(const MoveType&);
};

template<typename MoveType>
MultipleMovesProposer<MoveType>::MultipleMovesProposer(std::vector<Proposer<MoveType>*>& proposers,
    std::vector<double> moveWeights):
            m_proposers(proposers), m_moveWeights(moveWeights) {

    if (m_moveWeights.size() != m_proposers.size())
        throw std::invalid_argument("MultipleMovesProposer: Number of "
                "proposers isn't equal to the number of moveWeights.");
    if (m_proposers.size() == 0)
        throw std::invalid_argument("MultipleMovesProposer: No proposers given.");

    m_moveTypeDistribution = std::discrete_distribution<unsigned int>(moveWeights.begin(), moveWeights.end());
}

template<typename MoveType>
MoveType MultipleMovesProposer<MoveType>::operator()() {
    m_proposedMoveType = m_moveTypeDistribution(rng);
    return (*m_proposers[m_proposedMoveType])();
}

template<typename MoveType>
double MultipleMovesProposer<MoveType>::getProposalProb(const MoveType& move) const {
    return m_proposers[m_proposedMoveType]->getProposalProb(move);
}

template<typename MoveType>
void MultipleMovesProposer<MoveType>::updateProbabilities(const MoveType& move) {
    for (auto& proposer: m_proposers)
        proposer->updateProbabilities(move);
}

} // namespace FastMIDyNet

#endif
