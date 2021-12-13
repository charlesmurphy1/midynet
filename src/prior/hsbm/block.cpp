#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/exceptions.h"


using namespace std;


namespace FastMIDyNet{
//
// const std::vector<BlockIndex> BlockHierarchicalPrior::getBlockHierarchyOfIdx(BaseGraph::VertexIndex idx) const {
//     std::vector<BlockIndex> blockHierarchy;
//     for (auto b: m_state){
//         blockHierarchy.push_back(b[idx]);
//         idx = b[idx];
//     }
//     return blockHierarchy;
// }
//
// void BlockHierarchicalPrior::applyBlockMoveToVertexCounts(const NestedBlockMove& move){
//     for(size_t l=0; l<getLayerCount(); ++l){
//         auto m = move[l];
//
//         if (m.addedBlocks == 1) m_nestedVertexCountsInBlocks[l].push_back(0);
//         else if (m.addedBlocks == -1) m_nestedVertexCountsInBlocks[l].erase(m_nestedVertexCountsInBlocks[l].begin() + m.prevBlockIdx);
//
//         --m_nestedVertexCountsInBlocks[l][m.prevBlockIdx];
//         ++m_nestedVertexCountsInBlocks[l][m.nextBlockIdx];
//     }
// }
//
// vector<size_t> BlockHierarchicalPrior::computeVertexCountsInBlocks(const BlockSequence& state) {
//     size_t blockCount = *max_element(state.begin(), state.end()) + 1;
//
//     vector<size_t> vertexCount(blockCount, 0);
//     for (auto blockIdx: state) {
//         ++vertexCount[blockIdx];
//     }
//
//     return vertexCount;
// }
//
// void BlockHierarchicalPrior::checkBlockSequenceConsistencyWithBlockCount(const BlockSequence& blockSeq, size_t expectedBlockCount) {
//     size_t actualBlockCount = BlockHierarchicalPrior::computeBlockCount(blockSeq);
//     if (actualBlockCount != expectedBlockCount)
//         throw ConsistencyError("BlockHierarchicalPrior: blockSeq is inconsistent with expected block count.");
//
// }
//
// void BlockHierarchicalPrior::checkBlockSequenceConsistencyWithVertexCountsInBlocks(const BlockSequence& blockSeq, std::vector<size_t> expectedVertexCountsInBlocks) {
//     vector<size_t> actualVertexCountsInBlocks = computeVertexCountsInBlocks(blockSeq);
//     if (actualVertexCountsInBlocks.size() != expectedVertexCountsInBlocks.size())
//         throw ConsistencyError("BlockHierarchicalPrior: size of vertex count in blockSeq is inconsistent with expected block count.");
//
//     for (size_t i=0; i<actualVertexCountsInBlocks.size(); ++i){
//         auto x = actualVertexCountsInBlocks[i];
//         auto y = expectedVertexCountsInBlocks[i];
//         if (x != y){
//             throw ConsistencyError("BlockHierarchicalPrior: actual vertex count at index "
//             + to_string(i) + " is inconsistent with expected vertex count: "
//             + to_string(x) + " != " + to_string(y) + ".");
//         }
//     }
//
// }

} /* FastMIDyNet */
