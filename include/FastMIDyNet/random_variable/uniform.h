#ifndef FAST_MIDYNET_UNIFORM_RANDOM_VARIABLE_H
#define FAST_MIDYNET_UNIFORM_RANDOM_VARIABLE_H

#include "FastMIDyNet/random_variable/random_variable.hpp"


namespace FastMIDyNet{

class UniformRandomVariable: public RandomVariable<double, double, double>{

    protected:
        StateType a, b;

};

} // namespace FastMIDyNet

#endif
