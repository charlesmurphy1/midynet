#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/random_graph.hpp"

namespace FastMIDyNet {

void RandomGraph::_applyGraphMove(const GraphMove& move){
    for (auto edge: move.addedEdges){
        auto v = edge.first, u = edge.second;
        m_state.addEdgeIdx(v, u);
    }
    for (auto edge: move.removedEdges){
        auto v = edge.first, u = edge.second;
        if ( m_state.isEdgeIdx(u, v) )
            m_state.removeEdgeIdx(v, u);
        else
            throw std::runtime_error("Cannot remove non-existing edge (" + std::to_string(u) + ", " + std::to_string(v) + ").");
    }

}

}
