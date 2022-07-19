#ifndef FAST_MIDYNET_CM_DEGREE_H
#define FAST_MIDYNET_CM_DEGREE_H

#include <map>
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/erdosrenyi/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/maps.hpp"


namespace FastMIDyNet{

class DegreePrior: public Prior< DegreeSequence >{
protected:
    size_t m_size;
    EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
    DegreeCountsMap m_degreeCounts;

    void _samplePriors() override {
        m_edgeCountPriorPtr->sample();
    }

    void _applyGraphMove(const GraphMove& move) override;

    const double _getLogPrior() const override {
        return m_edgeCountPriorPtr->getLogJoint();
    }
    const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const { return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move); }

    // void onBlockCreation(const BlockMove&) override;

    void applyGraphMoveToState(const GraphMove&);
    void applyGraphMoveToDegreeCounts(const GraphMove&);
    void recomputeConsistentState() ;
public:
    /* Constructors */
    DegreePrior(size_t graphSize): m_size(graphSize){}
    DegreePrior(size_t graphSize, EdgeCountPrior& prior): m_size(graphSize){
            setEdgeCountPrior(prior);
        }
    DegreePrior(const DegreePrior& other){
        setEdgeCountPrior(*other.m_edgeCountPriorPtr);
    }
    virtual ~DegreePrior(){}
    const DegreePrior& operator=(const DegreePrior& other){
        setEdgeCountPrior(*other.m_edgeCountPriorPtr);
        return *this;
    }

    void setGraph(const MultiGraph&);
    // const MultiGraph& getGraph() const { return *m_graphPtr; }
    virtual void setState(const DegreeSequence&) override;
    static const DegreeCountsMap computeDegreeCounts(const std::vector<size_t>& degreeSequence);


    const size_t getSize() const { return m_size; }
    void setSize(size_t size) { m_size = size; }
    const size_t& getEdgeCount() const { return m_edgeCountPriorPtr->getState(); }
    const EdgeCountPrior& getEdgeCountPrior() const { return *m_edgeCountPriorPtr; }
    EdgeCountPrior& getEdgeCountPriorRef() const { return *m_edgeCountPriorPtr; }
    void setEdgeCountPrior(EdgeCountPrior& prior) {
        m_edgeCountPriorPtr = &prior;
        m_edgeCountPriorPtr->isRoot(false);
    }

    const size_t& getDegreeOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
    virtual const DegreeCountsMap& getDegreeCounts() const { return m_degreeCounts; }


    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;

    virtual void computationFinished() const override {
        m_isProcessed = false;
        m_edgeCountPriorPtr->computationFinished();
    }
    static void checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence&, size_t);
    static void checkDegreeSequenceConsistencyWithDegreeCounts(const DegreeSequence&, const DegreeCountsMap&);

    bool isSafe() const override {
        return (m_edgeCountPriorPtr != nullptr) and (m_edgeCountPriorPtr->isSafe());
    }
    void checkSelfConsistency() const override;
    virtual void checkSelfSafety() const override{
        if (m_edgeCountPriorPtr == nullptr)
            throw SafetyError("DegreePrior: unsafe prior since `m_edgeCountPriorPtr` is empty.");
        m_edgeCountPriorPtr->checkSafety();
    }


};


class DegreeDeltaPrior: public DegreePrior{
    DegreeSequence m_degreeSeq;

public:
    DegreeDeltaPrior(const DegreeSequence& degreeSeq):
        DegreePrior(degreeSeq.size()){ setState(degreeSeq); }

    DegreeDeltaPrior(const DegreeDeltaPrior& degreeDeltaPrior):
        DegreePrior(degreeDeltaPrior.m_size) {
            setState(degreeDeltaPrior.getState());
        }
    virtual ~DegreeDeltaPrior(){}
    const DegreeDeltaPrior& operator=(const DegreeDeltaPrior& other){
        this->setState(other.getState());
        return *this;
    }


    void setState(const DegreeSequence& degrees){
        m_size = degrees.size();
        m_degreeSeq = degrees;
        m_state = degrees;
    }
    void sampleState() override { };

    const double getLogLikelihood() const override { return 0.; }

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override;
    void checkSelfConsistency() const override { };
    void checkSelfSafety() const override {
        if (m_degreeSeq.size() == 0)
            throw SafetyError("DegreeDeltaPrior: unsafe prior since `m_degreeSeq` is empty.");
    }

    void computationFinished() const override { m_isProcessed = false; }

};

class DegreeUniformPrior: public DegreePrior{
    const double getLogLikelihoodFromEdgeCount(size_t edgeCount) const {
        return -logMultisetCoefficient(edgeCount, getSize());
    }
public:
    using DegreePrior::DegreePrior;
    void sampleState() override;

    const double getLogLikelihood() const override {
        return getLogLikelihoodFromEdgeCount(m_edgeCountPriorPtr->getState());
    }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const{
        int dE = move.addedEdges.size() - move.removedEdges.size();
        const size_t& E = m_edgeCountPriorPtr->getState();
        return getLogLikelihoodFromEdgeCount(E + dE) - getLogLikelihoodFromEdgeCount(E);
    }
};

class DegreeUniformHyperPrior: public DegreePrior{
public:

    using DegreePrior::DegreePrior;
    void sampleState() override;
    const double getLogLikelihood() const override;
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
};


} // FastMIDyNet

#endif
