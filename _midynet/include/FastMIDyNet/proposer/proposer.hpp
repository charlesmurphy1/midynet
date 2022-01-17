#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{


template<typename MoveType>
class Proposer{
    public:
        virtual MoveType proposeMove() const = 0;
        virtual void checkConsistency() const {};
        virtual void checkSafety() const {};
};

}

#endif
