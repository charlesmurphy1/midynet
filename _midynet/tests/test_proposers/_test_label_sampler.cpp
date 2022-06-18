// #include "gtest/gtest.h"
// #include "fixtures.hpp"
// #include "FastMIDyNet/proposer/sampler/label_sampler.h"
//
// namespace FastMIDyNet{
//
// class DummyLabelPairSampler: public LabelPairSampler{
// public:
//     using LabelPairSampler::LabelPairSampler;
//
//     const EdgeSampler& getEdgeSampler() const { return m_edgeSampler; }
//     const VertexUniformSampler& getVertexSampler() const { return m_vertexSampler; }
// };
//
// class TestLabelSampler: public ::testing::Test{
// public:
//
//     DummySBMGraph randomGraph = DummySBMGraph();
//     double shift = 1;
//     DummyLabelPairSampler sampler = DummyLabelPairSampler(shift);
//     BaseGraph::Edge existingEdge, abscentEdge;
//
//     void SetUp(){
//         seed(0);
//         randomGraph.sample();
//         sampler.setUp(randomGraph);
//         std::uniform_int_distribution<BaseGraph::VertexIndex> dist(0, randomGraph.getSize() - 1);
//
//         abscentEdge = getOrderedEdge({dist(rng), dist(rng)});
//         while(randomGraph.getGraph().getEdgeMultiplicityIdx(abscentEdge) != 0)
//             abscentEdge = getOrderedEdge({dist(rng), dist(rng)});
//
//         BaseGraph::VertexIndex vertex = dist(rng);
//         for (auto neighbor : randomGraph.getGraph().getNeighboursOfIdx(vertex)){
//             existingEdge = getOrderedEdge({vertex, neighbor.vertexIndex});
//             break;
//         }
//     }
// };
//
// TEST_F(TestLabelSampler, sample_returnLabelPairWithinGraph){
//     auto rs = sampler.sample();
//     EXPECT_GE(rs.first, 0);
//     EXPECT_LT(rs.first, randomGraph.getBlockCount());
//
//     EXPECT_GE(rs.second, 0);
//     EXPECT_LT(rs.second, randomGraph.getBlockCount());
// }
//
// TEST_F(TestLabelSampler, addEdge_forSomeEdge){
//     double weightBefore = sampler.getEdgeSampler().getEdgeWeight(abscentEdge);
//     sampler.onEdgeAddition(abscentEdge);
//     EXPECT_EQ(weightBefore + 1, sampler.getEdgeSampler().getEdgeWeight(abscentEdge));
// }
//
// TEST_F(TestLabelSampler, addEdge_forSomeSelfLoop){
//     abscentEdge.second = abscentEdge.first;
//     double weightBefore = sampler.getEdgeSampler().getEdgeWeight(existingEdge);
//     sampler.onEdgeAddition(existingEdge);
//     EXPECT_EQ(weightBefore + 1, sampler.getEdgeSampler().getEdgeWeight(existingEdge));
// }
//
// TEST_F(TestLabelSampler, removeEdge_forSomeEdge){
//     double weightBefore = sampler.getEdgeSampler().getEdgeWeight(existingEdge);
//     sampler.onEdgeRemoval(existingEdge);
//     EXPECT_EQ(weightBefore - 1, sampler.getEdgeSampler().getEdgeWeight(existingEdge));
// }
//
// TEST_F(TestLabelSampler, insertEdge_forSomeEdge){
//     sampler.onEdgeInsertion(abscentEdge, 10);
//     EXPECT_EQ(10, sampler.getEdgeSampler().getEdgeWeight(abscentEdge));
// }
//
// TEST_F(TestLabelSampler, eraseEdge_forSomeEdge){
//     double weight = sampler.getEdgeSampler().getEdgeWeight(existingEdge);
//     double oldWeight = sampler.onEdgeErasure(existingEdge);
//     EXPECT_EQ(weight, oldWeight);
//     EXPECT_EQ(sampler.getEdgeSampler().getEdgeWeight(existingEdge), 0);
//
// }
//
// TEST_F(TestLabelSampler, getLabelOfIdx_forSomeVertex_returnCorrectLabel){
//     EXPECT_EQ(sampler.getLabelOfIdx(5), randomGraph.getBlockOfIdx(5));
// }
//
// TEST_F(TestLabelSampler, getLabelOfIdx_forSomeEdge_returnCorrectLabels){
//     LabelPair rs = getOrderedPair<BlockIndex>({randomGraph.getBlockOfIdx(5), randomGraph.getBlockOfIdx(8)});
//     EXPECT_EQ(sampler.getLabelOfIdx({5, 8}), rs);
//
// }
//
// }
