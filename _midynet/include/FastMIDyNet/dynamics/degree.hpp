#ifndef FASTMIDYNET_DEGREE_DYNAMICS_H
#define FASTMIDYNET_DEGREE_DYNAMICS_H


#include "FastMIDyNet/dynamics/binary_dynamics.hpp"


namespace FastMIDyNet{

template<typename GraphPriorType=RandomGraph>
class DegreeDynamics: public BinaryDynamics<GraphPriorType> {
    double m_C;

    public:
        using BaseClass = BinaryDynamics<GraphPriorType>;

        DegreeDynamics(size_t numSteps, double C):
                BaseClass(numSteps, 0, 0, false, -1), m_C(C) {}
        DegreeDynamics(GraphPriorType& graphPrior, size_t numSteps, double C):
                BaseClass(graphPrior, numSteps, 0, 0, false, -1), m_C(C) { }

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            return (vertexNeighborState[0] + vertexNeighborState[1]) / m_C;
        }
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            return 1 - getActivationProb(vertexNeighborState);
        }
        const double getC() const { return m_C; }
        void setC(double C) { m_C = C; }

    };

} // namespace FastMIDyNet

#endif
