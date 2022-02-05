#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/proposer/edge_proposer/util.h"


namespace FastMIDyNet{

std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> getSubGraphOfLabelPair(const RandomGraph& randomGraph){

    const MultiGraph& graph = randomGraph.getGraph();
    const std::vector<BlockIndex>& blocks = randomGraph.getBlocks();
    size_t blockCount = randomGraph.getBlockCount();

    /* collecting edges */
    std::map<LabelPair, std::list<BaseGraph::Edge> > subEdges;
    for (auto vertex: graph){
        size_t r = blocks[vertex];
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
            size_t s = blocks[neighbor.vertexIndex];
            BaseGraph::Edge edge = getOrderedEdge({vertex, neighbor.vertexIndex});
            LabelPair labelPair = {r, s};
            if (subEdges.count(labelPair) == 0)
                subEdges.insert({labelPair, {edge}});
            else
                subEdges[labelPair].push_back(edge);
        }
    }

    /* constructing sub graphs */
    std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> subGraphs;
    for (size_t r=0; r<blockCount; ++r){
        for (size_t s=r; s<blockCount; ++s){
            LabelPair labelPair = {r, s};
            subGraphs.insert({labelPair, MultiGraph(graph.getSize())});
            for (auto e : subEdges[labelPair]){
                subGraphs[labelPair].setEdgeMultiplicityIdx(e, graph.getEdgeMultiplicityIdx(e));
            }
        }
    }

    for (auto g : subGraphs){
        std::cout << "Sugraph (" << g.first.first << ", " << g.first.second << ") : N=" << g.second.getSize() << ", E=" << g.second.getTotalEdgeNumber() << std::endl;
    }

    return subGraphs;
}

}
