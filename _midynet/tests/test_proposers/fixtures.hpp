#include "FastMIDyNet/types.h"

#include "FastMIDyNet/random_graph/prior/edge_count.h"
#include "FastMIDyNet/random_graph/prior/block_count.h"
#include "FastMIDyNet/random_graph/prior/block.h"
#include "FastMIDyNet/random_graph/prior/label_graph.h"
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
    size_t size;
    size_t edgeCount;
    size_t blockCount;

    BlockCountDeltaPrior blockCountPrior;
    BlockUniformPrior blockPrior;
    EdgeCountDeltaPrior edgeCountPrior;
    LabelGraphUniformPrior labelGraphPrior;

public:
    DummySBMGraph(size_t size=10, size_t edgeCount=25, size_t blockCount=3):
    StochasticBlockModelFamily(size),
    blockCountPrior(blockCount),
    blockPrior(size, blockCountPrior),
    edgeCountPrior(edgeCount),
    labelGraphPrior(edgeCountPrior, blockPrior)
     {
        setLabelGraphPrior(labelGraphPrior);
    }
    using StochasticBlockModelFamily::sample;
};

class DummyRestrictedSBMGraph: public StochasticBlockModelFamily{
    size_t size;
    size_t edgeCount;
    size_t blockCount;

    BlockCountDeltaPrior blockCountPrior;
    BlockUniformHyperPrior blockPrior;
    EdgeCountDeltaPrior edgeCountPrior;
    LabelGraphUniformPrior labelGraphPrior;

public:
    DummyRestrictedSBMGraph(size_t size=10, size_t edgeCount=25, size_t blockCount=3):
    StochasticBlockModelFamily(size),
    blockCountPrior(blockCount),
    blockPrior(size, blockCountPrior),
    edgeCountPrior(edgeCount),
    labelGraphPrior(edgeCountPrior, blockPrior)
     {
        setLabelGraphPrior(labelGraphPrior);
    }
    using StochasticBlockModelFamily::sample;
};

}
