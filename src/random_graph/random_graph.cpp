#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>

#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/types.h"

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
            m_state.removeEdgeIdx(v, u);
        }

    };
}
