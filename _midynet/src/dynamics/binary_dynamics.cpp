#include "FastMIDyNet/dynamics/binary_dynamics.h"
#include "FastMIDyNet/generators.h"


namespace FastMIDyNet {

const State BinaryDynamics::getRandomState() const {
    size_t N = m_randomGraphPtr->getSize();
    State randomState(N);
    if (m_numInitialActive > N)
        return Dynamics::getRandomState();

    auto indices = sampleUniformlySequenceWithoutReplacement(N, m_numInitialActive);
    for (auto i: indices)
        randomState[i] = 1;
    return randomState;
};

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
