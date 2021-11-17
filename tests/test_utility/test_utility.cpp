#include "gtest/gtest.h"

#include "FastMIDyNet/utility.h"


TEST(Utility, getDegree_graphWithSelfloop_returnExpectedDegrees) {
    FastMIDyNet::MultiGraph graph(4);
    graph.addEdgeIdx(0, 1);
    graph.addEdgeIdx(0, 0);
    graph.addEdgeIdx(2, 0);
    graph.addMultiedgeIdx(2, 2, 2);

    FastMIDyNet::DegreeSequence expectedDegrees = {4, 1, 5, 0};
    EXPECT_EQ(expectedDegrees, FastMIDyNet::getDegrees(graph));
    EXPECT_EQ(4, FastMIDyNet::getDegreeIdx(graph, 0));
    EXPECT_EQ(1, FastMIDyNet::getDegreeIdx(graph, 1));
    EXPECT_EQ(5, FastMIDyNet::getDegreeIdx(graph, 2));
    EXPECT_EQ(0, FastMIDyNet::getDegreeIdx(graph, 3));
}
