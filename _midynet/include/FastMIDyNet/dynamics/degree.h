#ifndef FASTMIDYNET_DEGREE_DYNAMICS_H
#define FASTMIDYNET_DEGREE_DYNAMICS_H


#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class DegreeDynamics: public BinaryDynamics {
    double m_C;

    public:
        DegreeDynamics(size_t numSteps, double C):
                BinaryDynamics(numSteps), m_C(C) {}
        DegreeDynamics(RandomGraph& random_graph, size_t numSteps, double C):
                BinaryDynamics(random_graph, numSteps), m_C(C) {}

        double getActivationProb(const VertexNeighborhoodState& neighborhood_state) const;
        double getDeactivationProb(const VertexNeighborhoodState& neighborhood_state) const;

    };

} // namespace FastMIDyNet

#endif
