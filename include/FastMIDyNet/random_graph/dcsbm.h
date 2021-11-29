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
#include "FastMIDyNet/utility/maps.h"
#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"

namespace FastMIDyNet{

class DegreeCorrectedStochasticBlockModelFamily: public StochasticBlockModelFamily{
protected:
    DegreePrior& m_degreePrior;
public:
    DegreeCorrectedStochasticBlockModelFamily(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior, DegreePrior& degreePrior):
    StochasticBlockModelFamily(blockPrior, edgeMatrixPrior), m_degreePrior(degreePrior) { }

    void sampleState () ;
    void samplePriors () ;

    const DegreeSequence& getDegreeSequence() const { return m_degreePrior.getState(); }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const  { return m_degreePrior.getDegreeCountsInBlocks(); }

    double getLogLikelihood() const;
    double getLogPrior() ;

    double getLogLikelihoodRatioEdgeTerm (const GraphMove& move) ;
    double getLogLikelihoodRatioAdjTerm (const GraphMove& move) ;
    double getLogLikelihoodRatio (const GraphMove& move) ;
    double getLogLikelihoodRatio (const BlockMove&) ;

    double getLogPriorRatio (const GraphMove&) ;
    double getLogPriorRatio (const BlockMove&) ;

    void applyMove (const GraphMove& move) ;
    void applyMove (const BlockMove&) ;

    void computationFinished(){
        m_blockPrior.computationFinished();
        m_edgeMatrixPrior.computationFinished();
        m_degreePrior.computationFinished();
    }

    static DegreeSequence getDegreeSequenceFromGraph(const MultiGraph&) ;
    static void checkGraphConsistencyWithDegreeSequence(const MultiGraph& graph, const DegreeSequence& degreeSeq) ;


    void checkSelfConsistency() ;

};

}// end FastMIDyNet
#endif
