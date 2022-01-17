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
    BlockPrior* m_blockPriorPtr = nullptr;
    EdgeMatrixPrior* m_edgeMatrixPriorPtr = nullptr;
    std::vector<size_t> m_degrees;
    std::vector<CounterMap<size_t>> m_degreeCounts;


public:
    StochasticBlockModelFamily(size_t graphSize): RandomGraph(graphSize) { }
    StochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
        RandomGraph(blockPrior.getSize()){
            setBlockPrior(blockPrior);
            setEdgeMatrixPrior(edgeMatrixPrior);
        }

    void sampleState () ;
    void samplePriors () ;


    void setState(const MultiGraph& state) {
        m_state = state;
        m_edgeMatrixPriorPtr->setGraph(m_state);
        m_degreeCounts = computeDegreeCountsInBlocks();
        m_degrees = m_state.getDegrees();
    }


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

    const BlockSequence& getBlocks() const override { return m_blockPriorPtr->getState(); }
    const size_t& getBlockCount() const { return m_blockPriorPtr->getBlockCount(); }
    const std::vector<size_t>& getVertexCountsInBlocks() const { return m_blockPriorPtr->getVertexCountsInBlocks(); }
    const EdgeMatrix& getEdgeMatrix() const { return m_edgeMatrixPriorPtr->getState(); }
    const std::vector<size_t>& getEdgeCountsInBlocks() const { return m_edgeMatrixPriorPtr->getEdgeCountsInBlocks(); }
    const size_t& getEdgeCount() const { return m_edgeMatrixPriorPtr->getEdgeCount(); }
    virtual const std::vector<size_t>& getDegrees() const { return m_degrees; }
    virtual const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const { return m_degreeCounts; }

    void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;
    void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge&, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>>&) const;
    void getDiffEdgeMatMapFromBlockMove(const BlockMove&, IntMap<std::pair<BlockIndex, BlockIndex>>&) const;

    virtual double getLogLikelihood() const;
    virtual double getLogPrior() const ;
    double getLogJoint() const { return getLogLikelihood() + getLogPrior(); }

    virtual double getLogLikelihoodRatioEdgeTerm (const GraphMove&) const ;
    virtual double getLogLikelihoodRatioAdjTerm (const GraphMove&) const ;

    virtual double getLogLikelihoodRatioFromGraphMove (const GraphMove&) const override;
    virtual double getLogLikelihoodRatioFromBlockMove (const BlockMove&) const override;

    virtual double getLogPriorRatioFromGraphMove (const GraphMove&) const override;
    virtual double getLogPriorRatioFromBlockMove (const BlockMove&) const override;

    virtual void applyGraphMove (const GraphMove&) override;
    virtual void applyBlockMove (const BlockMove&) override;

    virtual void computationFinished() const {
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
    }

    static EdgeMatrix getEdgeMatrixFromGraph(const MultiGraph&, const BlockSequence&) ;
    static void checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat);
    virtual void checkSelfConsistency() const ;
    virtual void checkSafety() const ;
};

}// end FastMIDyNet
#endif
