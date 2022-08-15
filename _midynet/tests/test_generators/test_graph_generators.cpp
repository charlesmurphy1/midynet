#include "gtest/gtest.h"

#include "../fixtures.hpp"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/generators.h"

namespace FastMIDyNet{

MultiGraph constructLabelGraph() {
    MultiGraph labelGraph(3);

    labelGraph.addMultiedgeIdx(0, 0, 2);
    labelGraph.addMultiedgeIdx(0, 1, 1);
    labelGraph.addMultiedgeIdx(0, 2, 3);
    labelGraph.addMultiedgeIdx(1, 2, 2);
    labelGraph.addMultiedgeIdx(2, 2, 3);
    return labelGraph;
}

static const MultiGraph LABEL_GRAPH = constructLabelGraph();
static const FastMIDyNet::BlockSequence VERTEX_BLOCKS = {
    0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2, 2, 2
};
static const FastMIDyNet::DegreeSequence DEGREES = {
    4, 2, 2, 0,
    2, 0, 1,
    0, 4, 4, 2, 1
};

static LabelGraph getLabelGraph(const FastMIDyNet::MultiGraph& graph, const std::vector<size_t>& vertexBlocks) {
    size_t blockNumber = 1;
    for (auto block: vertexBlocks)
        if (block >= blockNumber) blockNumber = block+1;

    LabelGraph labelGraph(blockNumber);
    size_t r, s;

    for (auto vertex: graph) {
        for (auto neighbor: graph.getNeighboursOfIdx(vertex)) {
            if (neighbor.vertexIndex < vertex)
                continue;
            r = vertexBlocks[vertex];
            s = vertexBlocks[neighbor.vertexIndex];
            labelGraph.addMultiedgeIdx(r, s, neighbor.label);
        }
    }
    return labelGraph;
}
static const size_t numberOfGeneratedGraphs = 10;

static const FastMIDyNet::DegreeSequence convertDegrees(const std::vector<size_t>& basegraphDegrees) {
    FastMIDyNet::DegreeSequence degrees;
    for (auto d: basegraphDegrees)
        degrees.push_back(d);
    return degrees;
}

TEST(TESTSampleRandomNeighbor, sampleRandomNeighbor_forMultipleSample_sampleAccordingToMultiplicity){
    seedWithTime();
    size_t numSamples = 1000;
    MultiGraph graph = getUndirectedHouseMultiGraph();
    graph.addMultiedgeIdx(1, 2, 2);
    graph.addMultiedgeIdx(1, 3, 1);
    graph.addMultiedgeIdx(1, 1, 2);

    CounterMap<BaseGraph::VertexIndex> counter;
    for(size_t i=0; i<numSamples; ++i){
        counter.increment(sampleRandomNeighbor(graph, 1, true));
    }

    for (const auto& c : counter){
        size_t m = ((c.first == 1) ? 2 : 1) * graph.getEdgeMultiplicityIdx(1, c.first);
        EXPECT_EQ(round(((double) c.second * 10) / numSamples), m);
    }
}


TEST(TestDCSBMGenerator, generateDCSBM_givenLabelGraphAndDegrees_generatedGraphsRespectLabelGraphAndDegrees) {
    for (size_t i=0; i<numberOfGeneratedGraphs; i++) {
        auto randomGraph = FastMIDyNet::generateDCSBM(VERTEX_BLOCKS, LABEL_GRAPH, DEGREES);
        EXPECT_EQ(LABEL_GRAPH, getLabelGraph(randomGraph, VERTEX_BLOCKS));
        EXPECT_EQ(DEGREES, convertDegrees(randomGraph.getDegrees()));
    }
}

TEST(TestSBMGenerator, generateStubLabeledSBM_givenLabelGraph_generatedGraphsRespectLabelGraph) {
    for (size_t i=0; i<numberOfGeneratedGraphs; i++) {
        auto randomGraph = FastMIDyNet::generateStubLabeledSBM(VERTEX_BLOCKS, LABEL_GRAPH);
        EXPECT_EQ(LABEL_GRAPH, getLabelGraph(randomGraph, VERTEX_BLOCKS));
    }
}

TEST(TestSBMGenerator, generateMultiGraphSBM_givenLabelGraph_generatedGraphsRespectLabelGraph) {
    for (size_t i=0; i<numberOfGeneratedGraphs; i++) {
        auto randomGraph = FastMIDyNet::generateMultiGraphSBM(VERTEX_BLOCKS, LABEL_GRAPH);
        EXPECT_EQ(LABEL_GRAPH, getLabelGraph(randomGraph, VERTEX_BLOCKS));
    }
}

}
