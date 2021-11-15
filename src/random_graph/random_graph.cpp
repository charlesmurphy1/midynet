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
        VertexIndex v_idx, u_idx;
        for (auto edge: move.edges_added){
            v_idx = edge.first;
            u_idx = edge.second;
            m_state.addEdgeIdx(v_idx, u_idx);
        }

    };
}
