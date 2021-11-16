#ifndef FAST_MIDYNET_SIS_DYNAMICS_H
#define FAST_MIDYNET_SIS_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/dynamics/binary_dynamics.h"


namespace FastMIDyNet{


class SISDynamic: public BinaryDynamics{

public:
        SISDynamic(RandomGraph& random_graph, RNG& rng, double infectionProb, double recoveryProb=0.5, double autoInfectionProb=1e-6) :
            BinaryDynamics(random_graph, rng), m_infectionProb(infectionProb), m_recoveryProb(recoveryProb), m_autoInfectionProb(autoInfectionProb)  { }

        double getActivationProb(const VertexNeighborhoodState& neighbor_state) const;
        double getDeactivationProb(const VertexNeighborhoodState& neighbor_state) const;

private:
        double m_infectionProb, m_recoveryProb, m_autoInfectionProb;
};

} // namespace FastMIDyNet

#endif
