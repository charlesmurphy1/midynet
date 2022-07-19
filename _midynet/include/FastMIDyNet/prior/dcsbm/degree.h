#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include <map>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/maps.hpp"


namespace FastMIDyNet{

class VertexLabeledDegreePrior: public BlockLabeledPrior< DegreeSequence >{
protected:
    EdgeMatrixPrior* m_edgeMatrixPriorPtr = nullptr;
    VertexLabeledDegreeCountsMap m_degreeCounts;

    void _samplePriors() override { m_edgeMatrixPriorPtr->sample(); }

    void _applyGraphMove(const GraphMove& move) override;
    void _applyLabelMove(const BlockMove& move) override;

    const double _getLogPrior() const override {
        return m_edgeMatrixPriorPtr->getLogJoint();
    }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const { return m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move); }
    const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const { return m_edgeMatrixPriorPtr->getLogJointRatioFromLabelMove(move); }

    // void onBlockCreation(const BlockMove&) override;

    void applyGraphMoveToState(const GraphMove&);
    void applyGraphMoveToDegreeCounts(const GraphMove&);
    void applyLabelMoveToDegreeCounts(const BlockMove&);
    void recomputeConsistentState() ;
public:
    using BlockLabeledPrior<DegreeSequence>::BlockLabeledPrior;
    /* Constructors */
    VertexLabeledDegreePrior(){}
    VertexLabeledDegreePrior(EdgeMatrixPrior& edgeMatrixPrior){
            setEdgeMatrixPrior(edgeMatrixPrior);
        }
    VertexLabeledDegreePrior(const VertexLabeledDegreePrior& other){
        setEdgeMatrixPrior(*other.m_edgeMatrixPriorPtr);
    }
    virtual ~VertexLabeledDegreePrior(){}
    const VertexLabeledDegreePrior& operator=(const VertexLabeledDegreePrior& other){
        setEdgeMatrixPrior(*other.m_edgeMatrixPriorPtr);
        return *this;
    }

    void samplePartition() { m_edgeMatrixPriorPtr->samplePartition(); recomputeConsistentState();}
    void setGraph(const MultiGraph&);
    // const MultiGraph& getGraph() const { return *m_graphPtr; }
    virtual void setState(const DegreeSequence&) override;
    void setPartition(const std::vector<BlockIndex>&) ;
    static const VertexLabeledDegreeCountsMap computeDegreeCounts(const std::vector<size_t>&,  const std::vector<BlockIndex>);

    const BlockPrior& getBlockPrior() const { return m_edgeMatrixPriorPtr->getBlockPrior(); }
    BlockPrior& getBlockPriorRef() const { return m_edgeMatrixPriorPtr->getBlockPriorRef(); }
    void setBlockPrior(BlockPrior& prior) const { m_edgeMatrixPriorPtr->setBlockPrior(prior); }

    const EdgeMatrixPrior& getEdgeMatrixPrior() const { return *m_edgeMatrixPriorPtr; }
    EdgeMatrixPrior& getEdgeMatrixPriorRef() const { return *m_edgeMatrixPriorPtr; }
    void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) {
        m_edgeMatrixPriorPtr = &edgeMatrixPrior;
        m_edgeMatrixPriorPtr->isRoot(false);
    }

    const BlockIndex& getDegreeOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    virtual const VertexLabeledDegreeCountsMap& getDegreeCounts() const { return m_degreeCounts; }


    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
    virtual const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const = 0;

    virtual void computationFinished() const override {
        m_isProcessed = false;
        m_edgeMatrixPriorPtr->computationFinished();
    }
    static void checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence&, size_t);
    static void checkDegreeSequenceConsistencyWithDegreeCounts(const DegreeSequence&, const BlockSequence&, const VertexLabeledDegreeCountsMap&);

    bool isSafe() const override {
        return (m_edgeMatrixPriorPtr != nullptr) and (m_edgeMatrixPriorPtr->isSafe());
    }
    void checkSelfConsistency() const override;
    virtual void checkSelfSafety() const override{
        if (m_edgeMatrixPriorPtr == nullptr)
            throw SafetyError("VertexLabeledDegreePrior: unsafe prior since `m_edgeMatrixPriorPtr` is empty.");
        m_edgeMatrixPriorPtr->checkSafety();
    }


};


class VertexLabeledDegreeDeltaPrior: public VertexLabeledDegreePrior{
    DegreeSequence m_degreeSeq;

public:
    VertexLabeledDegreeDeltaPrior(){}
    VertexLabeledDegreeDeltaPrior(const DegreeSequence& degreeSeq):
        VertexLabeledDegreePrior(){ setState(degreeSeq); }

    VertexLabeledDegreeDeltaPrior(const VertexLabeledDegreeDeltaPrior& degreeDeltaPrior):
        VertexLabeledDegreePrior() {
            setState(degreeDeltaPrior.getState());
        }
    virtual ~VertexLabeledDegreeDeltaPrior(){}
    const VertexLabeledDegreeDeltaPrior& operator=(const VertexLabeledDegreeDeltaPrior& other){
        this->setState(other.getState());
        return *this;
    }


    void setState(const DegreeSequence& degrees){
        m_degreeSeq = degrees;
        m_state = degrees;
    }
    void sampleState() override { };

    const double getLogLikelihood() const override { return 0.; }

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override{
        return (move.prevLabel != move.nextLabel) ? -INFINITY : 0.;
    }
    void checkSelfConsistency() const override { };
    void checkSelfSafety() const override {
        if (m_degreeSeq.size() == 0)
            throw SafetyError("DegreeDeltaPrior: unsafe prior since `m_degreeSeq` is empty.");
    }

    void computationFinished() const override { m_isProcessed = false; }

};

class VertexLabeledDegreeUniformPrior: public VertexLabeledDegreePrior{
public:
    using VertexLabeledDegreePrior::VertexLabeledDegreePrior;
    void sampleState() override;

    const double getLogLikelihood() const override;
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const;
};

class VertexLabeledDegreeUniformHyperPrior: public VertexLabeledDegreePrior{
public:

    using VertexLabeledDegreePrior::VertexLabeledDegreePrior;
    void sampleState() override;

    const double getLogLikelihood() const override;
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const;
};


} // FastMIDyNet

#endif
