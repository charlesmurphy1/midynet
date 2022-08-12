#include "FastMIDyNet/random_graph/sbm.h"

namespace FastMIDyNet{

std::vector<BlockIndex> getPlantedBlocks(std::vector<size_t> sizes){
    std::vector<BlockIndex> blocks;
    for (size_t r=0; r<sizes.size(); ++r){
        for (size_t i=0; i<sizes[r]; ++i){
            blocks.push_back(r);
        }
    }
    return blocks;
}

std::vector<BlockIndex> getPlantedBlocks(size_t size, size_t blockCount){
    std::vector<size_t> sizes(blockCount, size/blockCount);

    size_t sum=0;
    for (size_t s: sizes)
        sum+=s;
    size_t remainder = size - sum;
    for(size_t i=0; i<remainder; i++){
        ++sizes[i];
    }
    return getPlantedBlocks(sizes);
}

LabelGraph getPlantedLabelGraph(size_t blockCount, size_t edgeCount, double assortativity){
    LabelGraph labelGraph(blockCount);
    double a = (assortativity + 1.0) / 2.0;
    size_t e_in = ceil(edgeCount / blockCount * a);
    size_t e_out = floor(2 * edgeCount / (blockCount * (blockCount - 1)) * (1 - a));

    for (size_t r=0; r<blockCount; ++r){
        labelGraph.addMultiedgeIdx(r, r, e_in);
        for (size_t s=r+1; s<blockCount; ++s){
            labelGraph.addMultiedgeIdx(r, s, e_out);
        }
    }
    return labelGraph;
}

}
