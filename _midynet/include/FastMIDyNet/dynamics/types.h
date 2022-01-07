#ifndef FAST_MIDYNET_DYNAMICS_TYPES_H
#define FAST_MIDYNET_DYNAMICS_TYPES_H

#include <vector>

namespace FastMIDyNet{

typedef int VertexState;
typedef std::vector<VertexState> State;
typedef std::vector<VertexState> VertexStateSequence;
typedef std::vector<State> StateSequence;

typedef std::vector<VertexState> VertexNeighborhoodState;
typedef std::vector<VertexNeighborhoodState> VertexNeighborhoodStateSequence;
typedef std::vector<VertexNeighborhoodState> NeighborsState;
typedef std::vector<NeighborsState> NeighborsStateSequence;

}

#endif
