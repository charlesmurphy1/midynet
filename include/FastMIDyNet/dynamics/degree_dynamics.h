#ifndef FASTMIDYNET_DEGREE_DYNAMICS_H
#define FASTMIDYNET_DEGREE_DYNAMICS_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class DegreeDynamics: public BinaryDynamics {
    double m_C;

    public:
        DegreeDynamics(RandomGraph& random_graph, RNG& rng, double C):
                BinaryDynamics(random_graph, rng), m_C(C) {}

        double getActivationProb(const VertexNeighborhoodState& neighborhood_state) const;
        double getDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const;

    };

} // namespace FastMIDyNet

#endif
