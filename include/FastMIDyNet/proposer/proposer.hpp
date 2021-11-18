#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{


template<typename MoveType>
class Proposer{
    public:
        Proposer() { }
        std::pair<MoveType, double> operator()() {
            MoveType move = proposeMove();
            return { move, getProposalProb(move) };
        }
        virtual MoveType proposeMove() = 0;
        virtual double getLogProposalProbRatio(const MoveType&) const = 0;
        virtual void updateProbabilities(const MoveType&) = 0;
};

}

#endif
