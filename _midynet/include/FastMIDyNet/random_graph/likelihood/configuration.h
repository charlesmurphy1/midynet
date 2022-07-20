#ifndef FAST_MIDYNET_LIKELIHOOD_CONFIGURATION_H
#define FAST_MIDYNET_LIKELIHOOD_CONFIGURATION_H

#include "BaseGraph/types.h"
#include "FastMIDyNet/random_graph/likelihood/likelihood.hpp"
#include "FastMIDyNet/random_graph/prior/degree.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class ConfigurationModelLikelihood: public GraphLikelihoodModel{
public:

    const double getLogLikelihood() const override ;
    const double getLogLikelihoodRatioFromGraphMove (const GraphMove& move) const override ;
    DegreePrior** m_degreePriorPtrPtr = nullptr;
};


}

#endif
