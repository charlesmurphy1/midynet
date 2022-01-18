#ifndef FAST_MIDYNET_GRAPH_DISTANCE_H
#define FAST_MIDYNET_GRAPH_DISTANCE_H

#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class GraphDistance{
public:
    virtual double compute(const MultiGraph& graph1, const MultiGraph& graph2) const = 0;
};

class HammingDistance: public GraphDistance{

public:
    using GraphDistance::GraphDistance;
    double compute(const MultiGraph& graph1, const MultiGraph& graph2) const override;
};

}


#endif
