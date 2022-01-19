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
        explicit BinaryDynamics(size_t numSteps, bool normalizeCoupling=true):
            Dynamics(2, numSteps, normalizeCoupling) { }
        explicit BinaryDynamics(RandomGraph& randomGraph, size_t numSteps, bool normalizeCoupling=true):
            Dynamics(randomGraph, 2, numSteps, normalizeCoupling) { }
        const double getTransitionProb(VertexState prevVertexState,
                            VertexState nextVertexState,
                            VertexNeighborhoodState neighborhoodState
                        ) const override;

        virtual const double getActivationProb(const VertexNeighborhoodState& neighborState) const = 0;
        virtual const double getDeactivationProb(const VertexNeighborhoodState& neighborState) const = 0;
};

} // namespace FastMIDyNet

#endif
