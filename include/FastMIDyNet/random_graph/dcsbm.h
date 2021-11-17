#ifndef FAST_MIDYNET_DCSBM_H
#define FAST_MIDYNET_DCSBM_H

#include "BaseGraph/types.h"
#include "FastMIDyNet/prior/dcsbm/edge_matrix.h"
#include "FastMIDyNet/prior/dcsbm/block.h"
#include "FastMIDyNet/prior/dcsbm/degree.h"
#include "FastMIDyNet/random_graph/random_graph.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class DegreeCorrectedStochasticBlockModelFamily: public RandomGraph{
    EdgeMatrixPrior& m_edgeMatrixPrior;
    BlockPrior& m_blockPrior;
    DegreePrior& m_degreePrior;
public:
    DegreeCorrectedStochasticBlockModelFamily(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior, DegreePrior& degreePrior, RNG& rng):
    m_edgeMatrixPrior(edgeMatrixPrior),m_blockPrior(blockPrior), m_degreePrior(degreePrior), RandomGraph(blockPrior.getSize(), rng){ }

    void sampleState () ;

    const BlockSequence& getBlockSequence() { return m_blockPrior.getState(); }
    const BlockSequence& getBlockCount() { return m_blockPrior.getBlockCount(); }
    const EdgeMatrix& getEdgeMatrix() { return m_edgeMatrixPrior.getState(); }
    const EdgeMatrix& getEdgeCount() { return m_edgeMatrixPrior.getEdgeCount(); }
    const DegreeSequence& getDegreeSequence() { return m_degreePrior.getState(); }

    double getLogLikelihood() const;
    double getLogPrior() const ;
    double getLogJoint() const;

    double getLogLikelihoodRatio (const EdgeMove&, bool addition) const;
    double getLogLikelihoodRatio (const GraphMove& move) const{ return getLogLikelihoodRatio(move.edgesAdded, true) + getLogLikelihoodRatio(move.edgesRemoved, false); }
    double getLogLikelihoodRatio (const BaseGraph::VertexIndex&, const BlockIndex&, const BlockIndex&) const;
    double getLogLikelihoodRatio (const BlockMove&) const;
    double getLogPriorRatio (const GraphMove&) const;
    double getLogPriorRatio (const BlockMove&) const;
    double getLogJointRatio (const GraphMove& move) const { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }
    double getLogJointRatio (const BlockMove& move) const { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }

    static double getEr(const EdgeMatrix& edgeMatrix) ;

    void checkGraphConsistency(const MultiGraph& graph, BlockSequence blockSeq, EdgeMatrix edgeMat, DegreeSequence degreeSeq) ;
    void checkConsistency() ;

};

}// end FastMIDyNet
#endif
