#ifndef FAST_MIDYNET_POISSON_RANDOM_VARIABLE_H
#define FAST_MIDYNET_POISSON_RANDOM_VARIABLE_H

#include "FastMIDyNet/random_variable/random_variable.hpp"

namespace FastMIDyNet{

class PoissonRandomVariable: public RandomVariable<double, double, double>{

    protected:
        StateType mean;

};

} // namespace FastMIDyNet

#endif
