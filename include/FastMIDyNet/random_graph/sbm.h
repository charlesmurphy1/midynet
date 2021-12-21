#ifndef FAST_MIDYNET_SBM_H
#define FAST_MIDYNET_SBM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class StochasticBlockModelFamily: public RandomGraph{
protected:
    EdgeMatrixPrior& m_edgeMatrixPrior;
    BlockPrior& m_blockPrior;
public:
    StochasticBlockModelFamily(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
    m_blockPrior(blockPrior), m_edgeMatrixPrior(edgeMatrixPrior), RandomGraph(blockPrior.getSize()) {
        m_blockPrior.isRoot(false);
        m_edgeMatrixPrior.isRoot(false);
    }

    void sampleState () ;
    void samplePriors () ;

    void setState(const MultiGraph& state) { m_state = state; m_edgeMatrixPrior.setGraph(m_state); }

    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_blockPrior.getBlockOfIdx(idx); }
    const BlockSequence& getBlocks() const { return m_blockPrior.getState(); }
    const size_t& getBlockCount() const { return m_blockPrior.getBlockCount(); }
    const std::vector<size_t>& getVertexCountsInBlocks() const { return m_blockPrior.getVertexCountsInBlocks(); }
    const EdgeMatrix& getEdgeMatrix() const { return m_edgeMatrixPrior.getState(); }
    const std::vector<size_t>& getEdgeCountsInBlocks() const { return m_edgeMatrixPrior.getEdgeCountsInBlocks(); }
    const size_t& getEdgeCount() const { return m_edgeMatrixPrior.getEdgeCount(); }

    void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BlockIndex, BlockIndex>>&);
    void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>>&);
    void getDiffEdgeMatMapFromBlockMove(const BlockMove&, IntMap<std::pair<BlockIndex, BlockIndex>>&);

    virtual double getLogLikelihood() const;
    virtual double getLogPrior() ;
    double getLogJoint() { return getLogLikelihood() + getLogPrior(); }

    virtual double getLogLikelihoodRatioEdgeTerm (const GraphMove&) ;
    virtual double getLogLikelihoodRatioAdjTerm (const GraphMove&) ;

    virtual double getLogLikelihoodRatio (const GraphMove&) ;
    virtual double getLogLikelihoodRatio (const BlockMove&) ;

    virtual double getLogPriorRatio (const GraphMove&) ;
    virtual double getLogPriorRatio (const BlockMove&) ;

    double getLogJointRatio (const BlockMove& move) { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }

    virtual void applyMove (const GraphMove&);
    virtual void applyMove (const BlockMove&);

    virtual void computationFinished(){
        m_blockPrior.computationFinished();
        m_edgeMatrixPrior.computationFinished();
    }

    static EdgeMatrix getEdgeMatrixFromGraph(const MultiGraph&, const BlockSequence&) ;
    static void checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat);
    virtual void checkSelfConsistency() ;
};

}// end FastMIDyNet
#endif
