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
        VertexIndex v, u;
        for (auto edge: move.addedEdges){
            v = edge.first;
            u = edge.second;
            m_state.addEdgeIdx(v, u);
        }

    };
}
