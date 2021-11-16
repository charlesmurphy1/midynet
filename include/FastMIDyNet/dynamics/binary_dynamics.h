#ifndef FAST_MIDYNET_BINARY_DYNAMICS_H
#define FAST_MIDYNET_BINARY_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


class BinaryDynamics: public Dynamics{

    public:
        explicit BinaryDynamics(RandomGraph& randomGraph, RNG& rng):
            Dynamics(randomGraph, 2, rng) { }
        double getTransitionProb(
            VertexState prevVertexState,
            VertexState nextVertexState,
            VertexNeighborhoodState neighborhoodState
        ) const {
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

        virtual double getActivationProb(const VertexNeighborhoodState& neighboohState) const = 0;
        virtual double getDeactivationProb(const VertexNeighborhoodState& neighboohState) const = 0;
};

} // namespace FastMIDyNet

#endif
