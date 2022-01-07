#include "FastMIDyNet/types.h"


static FastMIDyNet::MultiGraph getUndirectedHouseMultiGraph(){
    //     /*
    //      * (0)     (1)
    //      * |||\   / | \
    //      * ||| \ /  |  \
    //      * |||  X   |  (4)
    //      * ||| / \  |  /
    //      * |||/   \ | /
    //      * (2)-----(3)-----(5)--
    //      *                   \__|
    //      *      (6)
    //      */
    // k = {4, 3, 5, 5, 2, 3, 0}
    FastMIDyNet::MultiGraph graph(7);
    graph.addMultiedgeIdx(0, 2, 3);
    graph.addEdgeIdx(0, 3);
    graph.addEdgeIdx(1, 2);
    graph.addEdgeIdx(1, 3);
    graph.addEdgeIdx(1, 4);
    graph.addEdgeIdx(2, 3);
    graph.addEdgeIdx(3, 4);
    graph.addEdgeIdx(3, 5);
    graph.addEdgeIdx(5, 5);

    return graph;

}
