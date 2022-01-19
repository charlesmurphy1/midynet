#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet {


const double BinaryDynamics::getTransitionProb(VertexState prevVertexState, VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState) const {
    double p;
    if ( prevVertexState == 0 ) {
        p = getActivationProb(neighborhoodState);
        if (nextVertexState == 0) return 1 - p;
        else return p;
    }
    else {
        p = getDeactivationProb(neighborhoodState);
        if (nextVertexState == 1) return 1 - p;
        else return p;
    }
};

} // namespace FastMIDyNet
