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
        explicit BinaryDynamics(RandomGraph& random_graph, RNG& rng):
            Dynamics(random_graph, 2, rng) { }
        const double getTransitionProb(
            VertexState prev_vertex_state,
            VertexState next_vertex_state,
            VertexNeighborhoodState neighborhood_state
        ) const {
            if ( prev_vertex_state == 0 ) {
                return getActivationProb(next_vertex_state, neighborhood_state);
            }
            else {
                return getDeactivationProb(next_vertex_state, neighborhood_state);
            }
        };

        virtual const double getActivationProb(VertexState next_vertex_state, VertexNeighborhoodState neighbooh_state) const = 0;
        virtual const double getDeactivationProb(VertexState next_vertex_state, VertexNeighborhoodState neighbooh_state) const = 0;
};

} // namespace FastMIDyNet

#endif
