#ifndef FAST_MIDYNET_DCSBM_H
#define FAST_MIDYNET_DCSBM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/prior/sbm/degree.h"
#include "FastMIDyNet/random_graph/sbm.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class DegreeCorrectedStochasticBlockModelFamily: public StochasticBlockModelFamily{
protected:
    DegreePrior* m_degreePriorPtr = nullptr;
public:
    DegreeCorrectedStochasticBlockModelFamily(size_t graphSize):
        StochasticBlockModelFamily(graphSize) { }
    DegreeCorrectedStochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior, DegreePrior& degreePrior):
        StochasticBlockModelFamily(graphSize, blockPrior, edgeMatrixPrior) {
            setDegreePrior(degreePrior);
        }

    void sampleState () ;
    void samplePriors () ;

    void setState(const MultiGraph& state) { m_state = state; m_degreePriorPtr->setGraph(m_state); }

    const DegreePrior& getDegreePrior() const { return *m_degreePriorPtr; }
    DegreePrior& getDegreePriorRef() const { return *m_degreePriorPtr; }
    virtual void setBlockPrior(BlockPrior& blockPrior) {
        StochasticBlockModelFamily::setBlockPrior(blockPrior);
        if (m_degreePriorPtr){
            m_degreePriorPtr->setBlockPrior(*m_blockPriorPtr);
        }
    }
    virtual void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) {
        StochasticBlockModelFamily::setEdgeMatrixPrior(edgeMatrixPrior);
        if (m_degreePriorPtr){
            m_degreePriorPtr->setBlockPrior(*m_blockPriorPtr);
            m_degreePriorPtr->setEdgeMatrixPrior(*m_edgeMatrixPriorPtr);
        }
    }
    virtual void setDegreePrior(DegreePrior& degreePrior) {
        m_degreePriorPtr = &degreePrior;
        m_degreePriorPtr->isRoot(false);
        m_degreePriorPtr->setBlockPrior(*m_blockPriorPtr);
        m_degreePriorPtr->setEdgeMatrixPrior(*m_edgeMatrixPriorPtr);
    }


    const BlockIndex& getDegreeOfIdx(BaseGraph::VertexIndex idx) const { return m_degreePriorPtr->getDegreeOfIdx(idx); }
    const DegreeSequence& getDegrees() const { return m_degreePriorPtr->getState(); }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const  { return m_degreePriorPtr->getDegreeCountsInBlocks(); }

    double getLogLikelihood() const;
    double getLogPrior() ;

    double getLogLikelihoodRatioEdgeTerm (const GraphMove&) ;
    double getLogLikelihoodRatioAdjTerm (const GraphMove&) ;

    double getLogLikelihoodRatio (const GraphMove&) ;
    double getLogLikelihoodRatio (const BlockMove&) ;

    double getLogPriorRatio (const GraphMove&) ;
    double getLogPriorRatio (const BlockMove&) ;

    void applyMove (const GraphMove&) ;
    void applyMove (const BlockMove&) ;

    void computationFinished() const {
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
        m_degreePriorPtr->computationFinished();
    }

    static DegreeSequence getDegreesFromGraph(const MultiGraph&) ;
    static void checkGraphConsistencyWithDegreeSequence(const MultiGraph&, const DegreeSequence&) ;


    void checkSelfConsistency() const ;
    virtual void checkSafety() const ;

};

}// end FastMIDyNet
#endif
