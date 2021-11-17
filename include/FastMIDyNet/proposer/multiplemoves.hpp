#ifndef FAST_MIDYNET_MULTIPLEMOVE_PROPOSER_H
#define FAST_MIDYNET_MULTIPLEMOVE_PROPOSER_H

#include "proposer.hpp"
#include "FastMIDyNet/types.h"
#include <random>


namespace FastMIDyNet {


template<typename MoveType>
class MultipleMovesProposer: public Proposer<MoveType> {
    RNG& m_rng;
    std::vector<Proposer<MoveType>*>& m_proposers;
    std::vector<double> m_moveProbabilities;
    unsigned int m_proposedMoveType=0;
    std::discrete_distribution<unsigned int> m_moveTypeDistribution;

    public:
        MultipleMovesProposer(std::vector<Proposer<MoveType>*>& proposers, std::vector<double> moveProbabilities, RNG& rng);
        MoveType operator()();
        double getProposalProb(const MoveType&) const;
        void updateProbabilities(const MoveType&);
};

template<typename MoveType>
MultipleMovesProposer<MoveType>::MultipleMovesProposer(std::vector<Proposer<MoveType>*>& proposers,
    std::vector<double> moveProbabilities, RNG& rng):
            m_proposers(proposers), m_rng(rng), m_moveProbabilities(moveProbabilities) {

    if (m_moveProbabilities.size() != m_proposers.size())
        throw std::invalid_argument("MultipleMovesProposer: Number of "
                "proposers isn't equal to the number of moveProbabilities.");
    if (m_proposers.size() == 0)
        throw std::invalid_argument("MultipleMovesProposer: No proposers given.");

    m_moveTypeDistribution = std::discrete_distribution<unsigned int>(moveProbabilities.begin(), moveProbabilities.end());
}

template<typename MoveType>
MoveType MultipleMovesProposer<MoveType>::operator()() {
    m_proposedMoveType = m_moveTypeDistribution(m_rng);
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
