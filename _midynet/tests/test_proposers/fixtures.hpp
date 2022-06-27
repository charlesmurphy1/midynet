#include "FastMIDyNet/types.h"

#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/random_graph/sbm.h"

namespace FastMIDyNet{
static MultiGraph getUndirectedHouseMultiGraph(){
    //     /*
    //      * (0)      (1)
    //      * ||| \   / | \
    //      * |||  \ /  |  \
    //      * |||   X   |  (4)
    //      * |||  / \  |  /
    //      * ||| /   \ | /
    //      * (2)------(3)-----(5)
    //      *
    //      *      (6)
    //      */
    MultiGraph graph(7);
    graph.addMultiedgeIdx(0, 2, 3);
    graph.addEdgeIdx(0, 3);
    graph.addEdgeIdx(1, 2);
    graph.addEdgeIdx(1, 3);
    graph.addEdgeIdx(1, 4);
    graph.addEdgeIdx(2, 3);
    graph.addEdgeIdx(3, 4);
    graph.addEdgeIdx(3, 5);

    return graph;

}

class DummySBMGraph: public StochasticBlockModelFamily{
private:
    size_t size;
    BlockCountDeltaPrior blockCount = {3};
    BlockUniformPrior blocks = {size, blockCount};
    EdgeCountDeltaPrior edgeCount = {250};
    EdgeMatrixUniformPrior edgeMatrix = {edgeCount, blocks};
public:
    DummySBMGraph(size_t size=100):StochasticBlockModelFamily(size){
        setBlockPrior(blocks);
        setEdgeMatrixPrior(edgeMatrix);
    }
};

class DummyRestrictedSBMGraph: public StochasticBlockModelFamily{
private:
    size_t size;
    BlockCountDeltaPrior blockCount = {3};
    BlockUniformHyperPrior blocks = {size, blockCount};
    EdgeCountDeltaPrior edgeCount = {250};
    EdgeMatrixUniformPrior edgeMatrix = {edgeCount, blocks};
public:
    DummyRestrictedSBMGraph(size_t size=100):StochasticBlockModelFamily(size){
        setBlockPrior(blocks);
        setEdgeMatrixPrior(edgeMatrix);
    }
};

}
