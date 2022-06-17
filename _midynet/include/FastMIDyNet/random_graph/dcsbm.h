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
    void samplePriors () override;

    void _applyGraphMove (const GraphMove&) override;
    void _applyBlockMove (const BlockMove&) override;
public:
    DegreeCorrectedStochasticBlockModelFamily(size_t graphSize):
        StochasticBlockModelFamily(graphSize) { }
    DegreeCorrectedStochasticBlockModelFamily(size_t graphSize, BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior, DegreePrior& degreePrior):
        StochasticBlockModelFamily(graphSize, blockPrior, edgeMatrixPrior) {
            setDegreePrior(degreePrior);
        }

    void sampleGraph () override;

    void setGraph(const MultiGraph& graph) override { m_graph = graph; m_degreePriorPtr->setGraph(m_graph); }

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


    const DegreeSequence& getDegrees() const override { return m_degreePriorPtr->getState(); }

    const double getLogLikelihood() const override ;
    const double getLogPrior() const override ;

    const double getLogLikelihoodRatioEdgeTerm (const GraphMove&) const override;
    const double getLogLikelihoodRatioAdjTerm (const GraphMove&) const override;

    const double getLogLikelihoodRatioFromGraphMove (const GraphMove&) const override;
    const double getLogLikelihoodRatioFromBlockMove (const BlockMove&) const override;

    const double getLogPriorRatioFromGraphMove (const GraphMove&) const override;
    const double getLogPriorRatioFromBlockMove (const BlockMove&) const override;

    void computationFinished() const override{
        m_isProcessed = false;
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
        m_degreePriorPtr->computationFinished();
    }


    static DegreeSequence getDegreesFromGraph(const MultiGraph&) ;
    static void checkGraphConsistencyWithDegreeSequence(const MultiGraph&, const DegreeSequence&) ;

    virtual bool isSafe() const override {
        return m_blockPriorPtr != nullptr and m_edgeMatrixPriorPtr != nullptr and m_degreePriorPtr != nullptr;
    }


    void checkSelfConsistency() const override;
    virtual void checkSelfSafety() const override;
    virtual const bool isCompatible(const MultiGraph& graph) const override{
        if (not StochasticBlockModelFamily::isCompatible(graph)) return false;
        auto degrees = getDegreesFromGraph(graph);
        return degrees == getDegrees();
    }

};

}// end FastMIDyNet
#endif
