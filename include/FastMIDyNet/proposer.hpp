#ifndef FAST_MIDYNET_PROPOSER_HPP
#define FAST_MIDYNET_PROPOSER_HPP

#include <random>
#include <cmath>

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{


template <typename MoveType>
class Proposer{
    public:
        explicit Proposer(RNGType p_rng): rng(p_rgn){ }
        virtual MoveType propose() = 0;
        virtual double moveLogProb(MoveType move, bool reversed=false) = 0;
        virtual double virtual(MoveType move) { return moveLogProb(move, true) - moveLogProb(move, false); }
        virtual void apply(MoveType move) = 0;
    protected:
        RNGType rng;
};

class IntProposer: protected Proposer<int>{
    public:
        double propose(){if (uniform_01(rng) > 0.5) {return 1;} else {return -1;}}
        double moveLogProb(double move, bool reversed=false) { return log(.5); }
        void apply(double move){ }
    private:
        std::uniform_real_distribution<double> uniform_01(0., 1.);
};

class DoubleProposer: protected Proposer<double>{
    public:
        explicit DoubleProposer(double p_step, RNGType p_rng): step(p_step), Proposer(p_rng){ }
        double getStep() { return step; }
        double setStep(double new_step) { step = new_step; }
        double propose(){ if (uniform_01(rng) > 0.5) {return step;} else {return -step;} }
        double moveLogProb(double move, bool reversed=false) { return log(.5); }
        void apply(double move){ }


    private:
        double step;
        std::uniform_real_distribution<double> uniform_01(0., 1.);
};

} // namespace FastMIDyNet

#endif
