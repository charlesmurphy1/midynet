#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree_count.h"
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

    void createBlock();
    void destroyBlock(const BlockIndex&);
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
    void setState(const DegreeSequence&) override;

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

    double getLogPrior() const override { return m_blockPriorPtr->getLogJoint() + m_edgeMatrixPriorPtr->getLogJoint(); }

    virtual double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
    virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const = 0;

    double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return m_blockPriorPtr->getLogJointRatioFromGraphMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromGraphMove(move); }
    double getLogPriorRatioFromBlockMove(const BlockMove& move) const { return m_blockPriorPtr->getLogJointRatioFromBlockMove(move) + m_edgeMatrixPriorPtr->getLogJointRatioFromBlockMove(move); }

    double getLogJointRatioFromGraphMove(const GraphMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move); }, 0);
    }

    double getLogJointRatioFromBlockMove(const BlockMove& move) {
        return processRecursiveFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0);
    }

    void applyGraphMoveToState(const GraphMove&);
    void applyBlockMoveToState(const BlockMove&){};
    void applyGraphMoveToDegreeCounts(const GraphMove&);
    void applyBlockMoveToDegreeCounts(const BlockMove&);
    void applyGraphMove(const GraphMove& move);
    void applyBlockMove(const BlockMove& move);

    void computationFinished() const override {
        m_isProcessed = false;
        m_blockPriorPtr->computationFinished();
        m_edgeMatrixPriorPtr->computationFinished();
    }
    static void checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence&, size_t);
    static void checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(const DegreeSequence&, const BlockSequence&, const std::vector<CounterMap<size_t>>&);
    void checkSelfConsistency() const override;
    virtual void checkSafety() const override{
        if (m_blockPriorPtr == nullptr)
            throw SafetyError("DegreePrior: unsafe prior since `m_blockPriorPtr` is empty.");

        if (m_edgeMatrixPriorPtr == nullptr)
            throw SafetyError("DegreePrior: unsafe prior since `m_edgeMatrixPriorPtr` is empty.");
    }


};


class DegreeDeltaPrior: public DegreePrior{
    DegreeSequence m_degreeSeq;
public:
    DegreeDeltaPrior(){}
    DegreeDeltaPrior(const DegreeSequence& degreeSeq,BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
        DegreePrior(blockPrior, edgeMatrixPrior){ setDegrees(degreeSeq); }

    DegreeDeltaPrior(const DegreeDeltaPrior& degreeDeltaPrior):
        DegreePrior() {
            setBlockPrior(degreeDeltaPrior.getBlockPriorRef());
            setEdgeMatrixPrior(degreeDeltaPrior.getEdgeMatrixPriorRef());
            setDegrees(degreeDeltaPrior.getState());
        }
    virtual ~DegreeDeltaPrior(){}
    const DegreeDeltaPrior& operator=(const DegreeDeltaPrior& other){
        this->setDegrees(other.getState());
        return *this;
    }


    void setDegrees(const DegreeSequence& degrees){
        m_degreeSeq = degrees;
        setState(degrees);
    }
    void sampleState() override { };
    void samplePriors() override { };

    double getLogLikelihood() const override { return 0.; }
    double getLogPrior() const override { return 0.; };

    double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const ;
    double getLogLikelihoodRatioFromBlockMove(const BlockMove& move) const { return 0.; }
    double getLogJointRatioFromBlockMove(const BlockMove& move) const {
        return processRecursiveConstFunction<double>( [&](){ return getLogLikelihoodRatioFromBlockMove(move); }, 0);
    }

    void checkSelfConsistency() const override { };
    void checkSafety() const override {
        DegreePrior::checkSafety();
        if (m_degreeSeq.size() == 0)
            throw SafetyError("DegreeDeltaPrior: unsafe prior since `m_degreeSeq` is empty.");
    }


};

class DegreeUniformPrior: public DegreePrior{
public:
    using DegreePrior::DegreePrior;
    void sampleState() override;

    double getLogLikelihood() const override;
    double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
};

class DegreeHyperPrior: public DegreePrior{

};


} // FastMIDyNet

#endif
