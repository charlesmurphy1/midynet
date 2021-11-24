#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "FastMIDyNet/types.h"
#include "FastMIDyNet/random_graph/random_graph.h"

using namespace std;
using namespace BaseGraph;

namespace FastMIDyNet {

    void RandomGraph::applyMove(const GraphMove& move){
        for (auto edge: move.addedEdges){
            auto v = edge.first, u = edge.second;
            m_state.addEdgeIdx(v, u);
        }
        for (auto edge: move.removedEdges){
            auto v = edge.first, u = edge.second;
            if ( m_state.isEdgeIdx(u, v) )
                m_state.removeEdgeIdx(v, u);
            else
                throw std::logic_error("Cannot remove non-existing edge ("+to_string(u)+","+to_string(v)+").");
        }

    };
}
