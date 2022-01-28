#ifndef FAST_MIDYNET_BINARY_DYNAMICS_H
#define FAST_MIDYNET_BINARY_DYNAMICS_H


#include <vector>
#include <map>
#include <unordered_map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/dynamics/dynamics.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet{

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};


class BinaryDynamics: public Dynamics{
private:
    bool m_cache;
    mutable std::unordered_map<std::pair<size_t,size_t>, double, pair_hash> m_activation_cache, m_deactivation_cache;
public:
    explicit BinaryDynamics(size_t numSteps, bool normalizeCoupling=true, bool cache=false):
        Dynamics(2, numSteps, normalizeCoupling), m_cache(cache) { }
    explicit BinaryDynamics(RandomGraph& randomGraph, size_t numSteps, bool normalizeCoupling=true, bool cache=false):
        Dynamics(randomGraph, 2, numSteps, normalizeCoupling), m_cache(cache) { }
    const double getTransitionProb(VertexState prevVertexState,
                        VertexState nextVertexState,
                        VertexNeighborhoodState neighborhoodState
                    ) const override;

    const double getActivationProb(const VertexNeighborhoodState& neighborState) const {
        if (not m_cache)
            return computeActivationProb(neighborState);
        std::pair<size_t, size_t> ns = {neighborState[0], neighborState[1]};
        if ( m_activation_cache.count(ns) == 0)
            m_activation_cache.insert({ns, computeActivationProb(neighborState)});

        return m_activation_cache[ns];
    }
    const double getDeactivationProb(const VertexNeighborhoodState& neighborState) const {
        if (not m_cache)
            return computeDeactivationProb(neighborState);
        std::pair<size_t, size_t> ns = {neighborState[0], neighborState[1]};
        if ( m_deactivation_cache.count(ns) == 0)
            m_deactivation_cache.insert({ns, computeDeactivationProb(neighborState)});
        return m_deactivation_cache[ns];
    }

    virtual const double computeActivationProb(const VertexNeighborhoodState& neighborState) const = 0;
    virtual const double computeDeactivationProb(const VertexNeighborhoodState& neighborState) const = 0;

    void clearCache(){ m_activation_cache.clear(); m_deactivation_cache.clear();}
};

} // namespace FastMIDyNet

#endif
