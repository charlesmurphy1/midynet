#ifndef FAST_MIDYNET_DCSBM_H
#define FAST_MIDYNET_DCSBM_H

#include <vector>

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
    DegreeCorrectedStochasticBlockModelFamily(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior, DegreePrior& degreePrior):
    m_edgeMatrixPrior(edgeMatrixPrior),m_blockPrior(blockPrior), m_degreePrior(degreePrior), RandomGraph(blockPrior.getSize()) { }

    void sampleState () ;

    const BlockSequence& getBlockSequence() const { return m_blockPrior.getState(); }
    const size_t& getBlockCount() const { return m_blockPrior.getBlockCount(); }
    const EdgeMatrix& getEdgeMatrix() const { return m_edgeMatrixPrior.getState(); }
    const size_t& getEdgeCount() const { return m_edgeMatrixPrior.getEdgeCount(); }
    const DegreeSequence& getDegreeSequence() const { return m_degreePrior.getState(); }

    double getLogLikelihood() const;
    double getLogPrior() const ;
    double getLogJoint() const;

    double getLogLikelihoodRatio (const EdgeMove&, bool addition) const;
    double getLogLikelihoodRatio (const GraphMove& move) const{ return getLogLikelihoodRatio(move.addedEdges, true) + getLogLikelihoodRatio(move.removedEdges, false); }
    double getLogLikelihoodRatio (const BaseGraph::VertexIndex&, const BlockIndex&, const BlockIndex&) const;
    double getLogLikelihoodRatio (const BlockMove&) const;
    double getLogPriorRatio (const GraphMove&) const;
    double getLogPriorRatio (const BlockMove&) const;
    double getLogJointRatio (const GraphMove& move) const { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }
    double getLogJointRatio (const BlockMove& move) const { return getLogLikelihoodRatio(move) + getLogPriorRatio(move); }

    static std::vector<size_t> getEr(const EdgeMatrix& edgeMatrix) ;
    static EdgeMatrix getEdgeMatrixFromGraph(const MultiGraph&, const BlockSequence&) ;
    static DegreeSequence getDegreeSequenceFromGraph(const MultiGraph&) ;
    static void checkGraphConsistencyWithEdgeMatrix(const MultiGraph& graph, const BlockSequence& blockSeq, const EdgeMatrix& expectedEdgeMat);
    static void checkGraphConsistencyWithDegreeSequence(const MultiGraph& graph, const DegreeSequence& degreeSeq) ;


    void checkSelfConsistency() ;

};

}// end FastMIDyNet
#endif
