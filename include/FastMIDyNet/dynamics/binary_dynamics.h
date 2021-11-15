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
        double getTransitionProb(
            VertexState prev_vertex_state,
            VertexState next_vertex_state,
            VertexNeighborhoodState neighborhood_state
        ) const {
            double p;
            if ( prev_vertex_state == 0 ) {
                p = getActivationProb(neighborhood_state);
                if (next_vertex_state == 0) return 1 - p;
                else return p;
            }
            else {
                p = getDeactivationProb(neighborhood_state);
                if (next_vertex_state == 1) return 1 - p;
                else return p;
            }
        };

        virtual double getActivationProb(const VertexNeighborhoodState& neighbooh_state) const = 0;
        virtual double getDeactivationProb(const VertexNeighborhoodState& neighbooh_state) const = 0;
};

} // namespace FastMIDyNet

#endif
