#ifndef FAST_MIDYNET_LABELED_EDGE_PROPOSER_H
#define FAST_MIDYNET_LABELED_EDGE_PROPOSER_H

#include <map>
#include "edge_proposer.h"
#include "FastMIDyNet/proposer/edge_sampler.h"

namespace FastMIDyNet{

using LabelPair = std::pair<BlockIndex, BlockIndex>;
class LabeledEdgeProposer: public EdgeProposer{
protected:
    const std::vector<BlockIndex>* m_labelsPtr = nullptr;
    EdgeSampler m_edgeSampler;
    bool m_keepLabels;

public:
    LabeledEdgeProposer(bool allowSelfLoops=true, bool allowMultiEdges=true, bool keepLabels=false):
        EdgeProposer(allowSelfLoops, allowMultiEdges), m_keepLabels(keepLabels){ }

    virtual void setUp( const RandomGraph& randomGraph ) {
        m_labelsPtr = &randomGraph.getBlocks(); setUpFromGraph(randomGraph.getGraph());
    }

    const bool& keepLabels(bool keep){ return m_keepLabels = keep; }
    const bool& keepLabels() const { return m_keepLabels; }

    virtual void onLabelCreation(const BlockMove& move);
    virtual void onLabelDeletion(const BlockMove& move);



};

}

#endif
