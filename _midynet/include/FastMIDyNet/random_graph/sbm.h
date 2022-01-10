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
    BlockPrior* m_blockPriorPtr = NULL;
    EdgeMatrixPrior* m_edgeMatrixPriorPtr = NULL;
public:
    StochasticBlockModelFamily(size_t graphSize): RandomGraph(graphSize) { }
    StochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
        RandomGraph(blockPrior.getSize()){
            setBlockPrior(blockPrior);
            setEdgeMatrixPrior(edgeMatrixPrior);
        }

    void sampleState () ;
    void samplePriors () ;


    void setState(const MultiGraph& state) { m_state = state; m_edgeMatrixPriorPtr->setGraph(m_state); }

    const BlockPrior& getBlockPrior() const { return *m_blockPriorPtr; }
    BlockPrior& getBlockPriorRef() const { return *m_blockPriorPtr; }
    virtual void setBlockPrior(BlockPrior& blockPrior) {
        m_blockPriorPtr = &blockPrior;
        m_blockPriorPtr->isRoot(false);
        m_blockPriorPtr->setSize(m_size);
        if (m_edgeMatrixPriorPtr)
            m_edgeMatrixPriorPtr->setBlockPrior(*m_blockPriorPtr);
    }

    const EdgeMatrixPrior& getEdgeMatrixPrior() const { return *m_edgeMatrixPriorPtr; }
    EdgeMatrixPrior& getEdgeMatrixPriorRef() const { return *m_edgeMatrixPriorPtr; }
    virtual void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) {
        m_edgeMatrixPriorPtr = &edgeMatrixPrior;
        m_edgeMatrixPriorPtr->isRoot(false);
        m_edgeMatrixPriorPtr->setBlockPrior(*m_blockPriorPtr);
    }

    const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_blockPriorPtr->getBlockOfIdx(idx); }
    const BlockSequence& getBlocks() const { return m_blockPriorPtr->getState(); }
    const size_t& getBlockCount() const { return m_blockPriorPtr->getBlockCount(); }
    const std::vector<size_t>& getVertexCountsInBlocks() const { return m_blockPriorPtr->getVertexCountsInBlocks(); }
    const EdgeMatrix& getEdgeMatrix() const { return m_edgeMatrixPriorPtr->getState(); }
    const std::vector<size_t>& getEdgeCountsInBlocks() const { return m_edgeMatrixPriorPtr->getEdgeCountsInBlocks(); }
    const size_t& getEdgeCount() const { return m_edgeMatrixPriorPtr->getEdgeCount(); }

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
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
    }

    static EdgeMatrix getEdgeMatrixFromGraph(const MultiGraph&, const BlockSequence&) ;
    static void checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat);
    virtual void checkSelfConsistency() ;
};

}// end FastMIDyNet
#endif
