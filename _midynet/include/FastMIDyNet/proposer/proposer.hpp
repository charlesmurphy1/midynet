#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/rv.hpp"

namespace FastMIDyNet{


template<typename MoveType>
class Proposer: public NestedRandomVariable{
    public:
        virtual ~Proposer(){}
        virtual MoveType proposeMove() const = 0;
        virtual void clear() {};
};

}

#endif
