#include "gtest/gtest.h"

#include "FastMIDyNet/utility/graph_util.h"
#include "FastMIDyNet/utility/functions.h"


TEST(GetDegree, graphWithSelfloop_returnExpectedDegrees) {
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

TEST(GetPoissonPMF, anyIntegerAndMeanCombination_returnCorrectLogPoissonPMF) {
    for (auto x: {0, 2, 10, 100})
        for (auto mu: {.0001, 1., 10., 1000.})
            EXPECT_DOUBLE_EQ(FastMIDyNet::logPoissonPMF(x, mu),
                                x*log(mu) - lgamma(x+1) - mu);
}
