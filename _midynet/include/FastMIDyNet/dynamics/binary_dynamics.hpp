#ifndef FAST_MIDYNET_BINARY_DYNAMICS_H
#define FAST_MIDYNET_BINARY_DYNAMICS_H


#include <vector>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.hpp"
#include "FastMIDyNet/dynamics/dynamics.hpp"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{

template <typename GraphPriorType=RandomGraph>
class BinaryDynamics: public Dynamics<GraphPriorType>{
private:
    size_t m_numInitialActive;
    double m_autoActivationProb;
    double m_autoDeactivationProb;
public:
    using BaseClass = Dynamics<GraphPriorType>;
    explicit BinaryDynamics(
            size_t numSteps,
            double autoActivationProb=0.0,
            double autoDeactivationProb=0.0,
            bool normalizeCoupling=true,
            size_t numInitialActive=-1):
        BaseClass(2, numSteps, normalizeCoupling),
        m_autoActivationProb(autoActivationProb),
        m_autoDeactivationProb(autoDeactivationProb),
        m_numInitialActive(numInitialActive) { }
    explicit BinaryDynamics(
            GraphPriorType& randomGraph,
            size_t numSteps,
            double autoActivationProb=0.0,
            double autoDeactivationProb=0.0,
            bool normalizeCoupling=true,
            size_t numInitialActive=-1):
        BaseClass(randomGraph, 2, numSteps, normalizeCoupling),
        m_autoActivationProb(autoActivationProb),
        m_autoDeactivationProb(autoDeactivationProb),
        m_numInitialActive(numInitialActive) { }
    const double getTransitionProb(VertexState prevVertexState,
                        VertexState nextVertexState,
                        VertexNeighborhoodState neighborhoodState
                    ) const override;

    const size_t getNumInitialActive() const { return m_numInitialActive; }
    void setNumInitialActive(size_t numInitialActive) {m_numInitialActive = numInitialActive; }
    const State getRandomState() const override;
    virtual const double getActivationProb(const VertexNeighborhoodState& neighborState) const = 0;
    virtual const double getDeactivationProb(const VertexNeighborhoodState& neighborState) const = 0;

    void setAutoActivationProb(double autoActivationProb){ m_autoActivationProb = autoActivationProb; }
    void setAutoDeactivationProb(double autoDeactivationProb){ m_autoDeactivationProb = autoDeactivationProb; }
    const double getAutoActivationProb() const { return m_autoActivationProb; }
    const double getAutoDeactivationProb() const { return m_autoDeactivationProb; }

};

template <typename GraphPriorType>
const State BinaryDynamics<GraphPriorType>::getRandomState() const {
    size_t N = BaseClass::m_graphPriorPtr->getSize();
    State randomState(N);
    if (m_numInitialActive > N)
        return Dynamics<GraphPriorType>::getRandomState();

    auto indices = sampleUniformlySequenceWithoutReplacement(N, m_numInitialActive);
    for (auto i: indices)
        randomState[i] = 1;
    return randomState;
};

template <typename GraphPriorType>
const double BinaryDynamics<GraphPriorType>::getTransitionProb(VertexState prevVertexState, VertexState nextVertexState,
        VertexNeighborhoodState neighborhoodState) const {
    double p;
    double transProb;
    if ( prevVertexState == 0 ) {
        p = (1 - m_autoActivationProb) * getActivationProb(neighborhoodState) + m_autoActivationProb;
        if (nextVertexState == 0) transProb = 1 - p;
        else transProb = p;
    }
    else {
        p = (1 - m_autoDeactivationProb) * getDeactivationProb(neighborhoodState) + m_autoDeactivationProb;
        if (nextVertexState == 1) transProb = 1 - p;
        else transProb = p;
    }

    return clipProb(transProb);
};

} // namespace FastMIDyNet

#endif
