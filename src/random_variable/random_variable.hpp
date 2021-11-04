#ifndef FAST_MIDYNET_RANDOM_VARIABLE_H
#define FAST_MIDYNET_RANDOM_VARIABLE_H

namespace FastMIDyNet{

template <typename StateType, typename StateMoveType, typename PriorMoveType>
class RandomVariable{

    public:
        StateType getState() { return state; }
        StateType setState(StateType new_state) { state = new_state; }
        virtual StateType sample() = 0;
        virtual double loglikelihood(StateType state) = 0;

        virtual MoveType proposeStateMove() = 0;
        virtual double virtualStateMove(StateMoveType move) = 0;
        virtual void applyStateMove(StateMoveType move) = 0;

        virtual MoveType proposePriorMove() = 0;
        virtual double virtualPriorMove(PriorMoveType move) = 0;
        virtual void applyPriorMove(PriorMoveType move) = 0;

    protected:
        StateType state;

        void checkConsistency() {}
};

} // namespace FastMIDyNet

#endif
