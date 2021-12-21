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
    DegreePrior& m_degreePrior;
public:
    DegreeCorrectedStochasticBlockModelFamily(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior, DegreePrior& degreePrior):
    StochasticBlockModelFamily(blockPrior, edgeMatrixPrior), m_degreePrior(degreePrior) { m_degreePrior.isRoot(false); }

    void sampleState () ;
    void samplePriors () ;

    void setState(const MultiGraph& state) { m_state = state; m_degreePrior.setGraph(m_state); }

    const BlockIndex& getDegreeOfIdx(BaseGraph::VertexIndex idx) const { return m_degreePrior.getDegreeOfIdx(idx); }
    const DegreeSequence& getDegrees() const { return m_degreePrior.getState(); }
    const std::vector<CounterMap<size_t>>& getDegreeCountsInBlocks() const  { return m_degreePrior.getDegreeCountsInBlocks(); }

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

    void computationFinished(){
        m_blockPrior.computationFinished();
        m_edgeMatrixPrior.computationFinished();
        m_degreePrior.computationFinished();
    }

    static DegreeSequence getDegreesFromGraph(const MultiGraph&) ;
    static void checkGraphConsistencyWithDegreeSequence(const MultiGraph&, const DegreeSequence&) ;


    void checkSelfConsistency() ;

};

}// end FastMIDyNet
#endif
