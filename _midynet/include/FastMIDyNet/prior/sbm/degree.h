#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include <map>
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/maps.hpp"


namespace FastMIDyNet{

typedef std::vector<CounterMap<size_t>> DegreeCountsMap;

class DegreePrior: public Prior< DegreeSequence >{
protected:
    BlockPrior* m_blockPriorPtr = nullptr;
    EdgeMatrixPrior* m_edgeMatrixPriorPtr = nullptr;
    DegreeCountsMap m_degreeCountsInBlocks;
    const MultiGraph* m_graph;

    void _applyGraphMove(const GraphMove& move) override;
    void _applyBlockMove(const BlockMove& move) override;

    const double _getLogJointRatioFromGraphMove(const GraphMove& move) const override{
        return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move);
    }
    const double _getLogJointRatioFromBlockMove(const BlockMove& move) const override {
        return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move);
    }

    void onBlockCreation(const BlockMove&) override;

    void applyGraphMoveToState(const GraphMove&);
    void applyGraphMoveToDegreeCounts(const GraphMove&);
    void applyBlockMoveToDegreeCounts(const BlockMove&);
public:
    using Prior<DegreeSequence>::Prior;
    /* Constructors */
    DegreePrior(){}
    DegreePrior(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior){
            setBlockPrior(blockPrior);
            setEdgeMatrixPrior(edgeMatrixPrior);
        }
    DegreePrior(const DegreePrior& other){
        setBlockPrior(*other.m_blockPriorPtr);
        setEdgeMatrixPrior(*other.m_edgeMatrixPriorPtr);
    }
    virtual ~DegreePrior(){}
    const DegreePrior& operator=(const DegreePrior& other){
        setBlockPrior(*other.m_blockPriorPtr);
        setEdgeMatrixPrior(*other.m_edgeMatrixPriorPtr);
        return *this;
    }

    void setGraph(const MultiGraph&);
    const MultiGraph& getGraph() const { return *m_graph; }
    virtual void setState(const DegreeSequence&) override;

    const BlockPrior& getBlockPrior() const { return *m_blockPriorPtr; }
    BlockPrior& getBlockPriorRef() const { return *m_blockPriorPtr; }
    void setBlockPrior(BlockPrior& blockPrior) { m_blockPriorPtr = &blockPrior; m_blockPriorPtr->isRoot(false);}

    const EdgeMatrixPrior& getEdgeMatrixPrior() const { return *m_edgeMatrixPriorPtr; }
    EdgeMatrixPrior& getEdgeMatrixPriorRef() const { return *m_edgeMatrixPriorPtr; }
    void setEdgeMatrixPrior(EdgeMatrixPrior& edgeMatrixPrior) {
        m_edgeMatrixPriorPtr = &edgeMatrixPrior; m_edgeMatrixPriorPtr->isRoot(false);
    }

    const BlockIndex& getDegreeOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    static DegreeCountsMap computeDegreeCountsInBlocks(const DegreeSequence& degreeSeq, const BlockSequence& blockSeq);
    virtual const DegreeCountsMap& getDegreeCountsInBlocks() const { return m_degreeCountsInBlocks; }


    void samplePriors() override { m_blockPriorPtr->sample(); m_edgeMatrixPriorPtr->sample(); }

    const double getLogPrior() const override { return m_blockPriorPtr->getLogJoint() + m_edgeMatrixPriorPtr->getLogJoint(); }

    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
    virtual const double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const = 0;
    virtual const double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return m_blockPriorPtr->getLogJointRatioFromGraphMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move); }
    virtual const double getLogPriorRatioFromBlockMove(const BlockMove& move) const { return m_blockPriorPtr->getLogJointRatioFromBlockMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromBlockMove(move); }

    virtual void computationFinished() const override {
        m_isProcessed = false;
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
    }
    static void checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence&, size_t);
    static void checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(const DegreeSequence&, const BlockSequence&, const std::vector<CounterMap<size_t>>&);
    void checkSelfConsistency() const override;
    virtual void checkSelfSafety() const override{
        if (m_blockPriorPtr == nullptr)
            throw SafetyError("DegreePrior: unsafe prior since `m_blockPriorPtr` is empty.");

        if (m_edgeMatrixPriorPtr == nullptr)
            throw SafetyError("DegreePrior: unsafe prior since `m_edgeMatrixPriorPtr` is empty.");
        m_blockPriorPtr->checkSafety();
        m_edgeMatrixPriorPtr->checkSafety();
    }


};


class DegreeDeltaPrior: public DegreePrior{
    DegreeSequence m_degreeSeq;

public:
    DegreeDeltaPrior(){}
    DegreeDeltaPrior(const DegreeSequence& degreeSeq):
        DegreePrior(){ setState(degreeSeq); }

    DegreeDeltaPrior(const DegreeDeltaPrior& degreeDeltaPrior):
        DegreePrior() {
            setState(degreeDeltaPrior.getState());
        }
    virtual ~DegreeDeltaPrior(){}
    const DegreeDeltaPrior& operator=(const DegreeDeltaPrior& other){
        this->setState(other.getState());
        return *this;
    }


    void setState(const DegreeSequence& degrees){
        m_degreeSeq = degrees;
        m_state = degrees;
    }
    void sampleState() override { };
    void samplePriors() override { };

    const double getLogLikelihood() const override { return 0.; }
    const double getLogPrior() const override { return 0.; };

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override;
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const override{
        return (move.prevBlockIdx != move.nextBlockIdx) ? -INFINITY : 0.;
    }
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0.; }
    const double getLogPriorRatioFromBlockMove(const BlockMove& move) const override{ return 0.; }

    void checkSelfConsistency() const override { };
    void checkSelfSafety() const override {
        if (m_degreeSeq.size() == 0)
            throw SafetyError("DegreeDeltaPrior: unsafe prior since `m_degreeSeq` is empty.");
    }

    void computationFinished() const override { m_isProcessed = false; }

};

class DegreeUniformPrior: public DegreePrior{
public:
    using DegreePrior::DegreePrior;
    void sampleState() override;

    const double getLogLikelihood() const override;
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
};

class DegreeUniformHyperPrior: public DegreePrior{
public:

    using DegreePrior::DegreePrior;
    void sampleState() override;

    const double getLogLikelihood() const override;
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    const double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
};


} // FastMIDyNet

#endif
