

// void BlockPrior::relabelBlock(BlockIndex blockIdx, BlockIndex newBlockIdx){
//     BlockSequence newState = m_state;
//     for (size_t i = 0; i < getSize(); ++i) {
//         if (m_state[i] == blockIdx) {
//             newState[i] = newBlockIdx;
//         }
//     }
//     setState(newState);
// }
//
// void BlockPrior::swapBlocks(map<BlockIndex, BlockIndex> swap){
//     BlockSequence newState = m_state;
//     for (size_t i = 0; i < getSize(); ++i) {
//         if (swap.count(m_state[i]) > 0) newState[i] = swap[m_state[i]];
//     }
//     setState(newState);
// }
//
// void BlockPrior::reduceBlocks(){
//     vector<BlockIndex> emptyBlocks;
//     for ( size_t blockIdx = 0; blockIdx < getBlockCount(); ++blockIdx ){
//         if (m_vertexCountsInBlocks[blockIdx] == 0) {
//             emptyBlocks.push_back(blockIdx);
//         }
//     }
//
//     for (size_t blockIdx = getBlockCount() - 1; blockIdx == 0; --blockIdx){
//         BlockIndex newBlockIdx = emptyBlocks.back();
//         if (m_vertexCountsInBlocks[blockIdx] > 0 && newBlockIdx > blockIdx){
//             relabelBlock(blockIdx, newBlockIdx);
//             emptyBlocks.pop_back();
//         }
//         if (emptyBlocks.empty()) break;
//     }
// }
//
// void BlockPrior::shuffleBlocks(){
//     size_t numBlocks = getBlockCount() ;
//     vector<BlockIndex> blockIdx = sampleUniformlySequenceWithoutReplacement(numBlocks, numBlocks - (numBlocks % 2));
//     vector<size_t> newVertexCountsInBlocks(numBlocks, 0);
//     map<BlockIndex, BlockIndex> swap;
//     while(swap.size() > 0){
//         auto r = blockIdx.back(); blockIdx.pop_back();
//         auto s = blockIdx.back(); blockIdx.pop_back();
//         swap.insert({r, s});
//         swap.insert({s, r});
//     }
//     swapBlocks(swap);
// }
