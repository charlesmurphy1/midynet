#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{


template<typename MoveType>
class Proposer{
    public:
        Proposer() { }
        virtual MoveType operator()() = 0;
        virtual double getProposalProb(const MoveType&) const = 0;
        virtual void updateProbabilities(const MoveType&) = 0;
};

}

#endif
