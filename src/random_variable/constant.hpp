#ifndef FAST_MIDYNET_POISSON_RANDOM_VARIABLE_H
#define FAST_MIDYNET_POISSON_RANDOM_VARIABLE_H

#include "FastMIDyNet/random_variable/random_variable.hpp"

namespace FastMIDyNet{

class EmptyPrior{};

template <typename StateType>
class ConstantVariable: public RandomVariable<StateType, StateType, EmptyPrior>{

        public:
            StateType getState() { return state; }
            StateType setState(StateType new_state) { }
            virtual StateType sample() = 0;
            virtual double loglikelihood(StateType state) { if (new_state != state) { return 0.; } else { return 1.; } }

            virtual StateType proposeStateMove() { return state; }
            virtual double virtualStateMove(StateType new_state) { if (new_state != state) { return 0.; } else { return 1.; } }
            virtual void applyStateMove(StateType new_state) { }

            virtual EmptyPrior proposePriorMove() { return state; }
            virtual double virtualPriorMove(EmptyPrior move) { return 1; }
            virtual void applyPriorMove(EmptyPrior move) { }

        protected:
            StateType state;

            void checkConsistency() {}
    };

} // namespace FastMIDyNet

#endif
