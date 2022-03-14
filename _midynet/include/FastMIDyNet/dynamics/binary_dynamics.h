#ifndef FAST_MIDYNET_BINARY_DYNAMICS_H
#define FAST_MIDYNET_BINARY_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{


class BinaryDynamics: public Dynamics{
private:
    size_t m_numInitialActive;
public:
    explicit BinaryDynamics(size_t numSteps, bool normalizeCoupling=true, size_t numInitialActive=-1):
        Dynamics(2, numSteps, normalizeCoupling), m_numInitialActive(numInitialActive) { }
    explicit BinaryDynamics(
        RandomGraph& randomGraph, size_t numSteps, bool normalizeCoupling=true, size_t numInitialActive=-1
    ): Dynamics(randomGraph, 2, numSteps, normalizeCoupling), m_numInitialActive(numInitialActive) { }
    const double getTransitionProb(VertexState prevVertexState,
                        VertexState nextVertexState,
                        VertexNeighborhoodState neighborhoodState
                    ) const override;

    const size_t getNumInitialActive() const { return m_numInitialActive; }
    void setNumInitialActive(size_t numInitialActive) {m_numInitialActive = numInitialActive; }
    const State getRandomState() const override;
    virtual const double getActivationProb(const VertexNeighborhoodState& neighborState) const = 0;
    virtual const double getDeactivationProb(const VertexNeighborhoodState& neighborState) const = 0;

};

} // namespace FastMIDyNet

#endif
