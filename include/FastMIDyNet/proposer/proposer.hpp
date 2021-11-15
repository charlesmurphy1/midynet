#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{


template <typename T>
class Proposer{
    public:
        Proposer() { }
        virtual T operator()() const = 0;
        virtual double getProposalProb(const T&) = 0;
        virtual void applyMove(const T&) = 0;
};

}

#endif
