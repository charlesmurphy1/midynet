#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"


namespace FastMIDyNet{

class EdgeMatrixPrior: public BlockLabeledPrior< MultiGraph >{
    protected:
        EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
        BlockPrior* m_blockPriorPtr = nullptr;
        CounterMap<size_t> m_edgeCounts;
        const MultiGraph* m_graphPtr;

        void _applyGraphMove(const GraphMove& move) override {
            m_edgeCountPriorPtr->applyGraphMove(move);
            m_blockPriorPtr->applyGraphMove(move);
            applyGraphMoveToState(move);
        }
        void _applyLabelMove(const BlockMove& move) override {
            m_edgeCountPriorPtr->applyLabelMove(move);
            m_blockPriorPtr->applyLabelMove(move);
            applyLabelMoveToState(move);
        }

        const double _getLogJointRatioFromGraphMove(const GraphMove& move) const override {
            return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move);
        }

        const double _getLogJointRatioFromLabelMove(const BlockMove& move) const override{
            return getLogLikelihoodRatioFromLabelMove(move) + getLogPriorRatioFromLabelMove(move);
        }

        void applyGraphMoveToState(const GraphMove&);
        void applyLabelMoveToState(const BlockMove&);
    public:
        EdgeMatrixPrior() {}
        EdgeMatrixPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior){
                setEdgeCountPrior(edgeCountPrior);
                setBlockPrior(blockPrior);
            }
        EdgeMatrixPrior(const EdgeMatrixPrior& other){
            setEdgeCountPrior(*other.m_edgeCountPriorPtr);
            setBlockPrior(*other.m_blockPriorPtr);
        }
        const EdgeMatrixPrior& operator=(const EdgeMatrixPrior& other){
            setEdgeCountPrior(*other.m_edgeCountPriorPtr);
            setBlockPrior(*other.m_blockPriorPtr);
            return *this;
        }

        const EdgeCountPrior& getEdgeCountPrior() const{ return *m_edgeCountPriorPtr; }
        EdgeCountPrior& getEdgeCountPriorRef() const{ return *m_edgeCountPriorPtr; }
        void setEdgeCountPrior(EdgeCountPrior& edgeCountPrior) { m_edgeCountPriorPtr = &edgeCountPrior; m_edgeCountPriorPtr->isRoot(false);}
        const BlockPrior& getBlockPrior() const{ return *m_blockPriorPtr; }
        BlockPrior& getBlockPriorRef() const{ return *m_blockPriorPtr; }
        void setBlockPrior(BlockPrior& blockPrior) {
            m_blockPriorPtr = &blockPrior;
            m_blockPriorPtr->isRoot(false);
        }

        void setGraph(const MultiGraph& graph);
        const MultiGraph& getGraph() { return *m_graphPtr; }
        void setState(const MultiGraph&) override;

        const size_t& getEdgeCount() const { return m_edgeCountPriorPtr->getState(); }
        const CounterMap<size_t>& getEdgeCounts() const { return m_edgeCounts; }


        void samplePriors() override { m_edgeCountPriorPtr->sample(); m_blockPriorPtr->sample(); }

        const double getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint() + m_blockPriorPtr->getLogJoint(); }

        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
        virtual const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const = 0;

        virtual const double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move) + m_blockPriorPtr->getLogJointRatioFromGraphMove(move); }
        virtual const double getLogPriorRatioFromLabelMove(const BlockMove& move) const { return m_edgeCountPriorPtr->getLogJointRatioFromLabelMove(move) + m_blockPriorPtr->getLogJointRatioFromLabelMove(move); }


        bool isSafe() const override {
            return (m_blockPriorPtr != nullptr) and (m_blockPriorPtr->isSafe())
               and (m_edgeCountPriorPtr != nullptr) and (m_edgeCountPriorPtr->isSafe());
        }
        void computationFinished() const override {
            m_isProcessed = false;
            m_blockPriorPtr->computationFinished();
            m_edgeCountPriorPtr->computationFinished();
        }
        void checkSelfConsistencywithGraph() const;
        void checkSelfConsistency() const override;

        void checkSelfSafety()const override{
            if (m_blockPriorPtr == nullptr)
                throw SafetyError("EdgeMatrixPrior: unsafe prior since `m_blockPriorPtr` is empty.");
            if (m_edgeCountPriorPtr == nullptr)
                throw SafetyError("EdgeMatrixPrior: unsafe prior since `m_edgeCountPriorPtr` is empty.");
            m_blockPriorPtr->checkSafety();
            m_edgeCountPriorPtr->checkSafety();
        }
};

class EdgeMatrixDeltaPrior: public EdgeMatrixPrior{
public:
    MultiGraph m_edgeMatrix;
    EdgeCountDeltaPrior m_edgeCountDeltaPrior;

public:
    EdgeMatrixDeltaPrior(){}
    EdgeMatrixDeltaPrior(const MultiGraph& edgeMatrix) {
        setState(edgeMatrix);
        m_edgeCountDeltaPrior.setState(edgeMatrix.getTotalEdgeNumber());
    }
    EdgeMatrixDeltaPrior(const MultiGraph& edgeMatrix, EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior):
        EdgeMatrixPrior(edgeCountPrior, blockPrior){ setState(edgeMatrix); }

    EdgeMatrixDeltaPrior(const EdgeMatrixDeltaPrior& edgeMatrixDeltaPrior):
        EdgeMatrixPrior(edgeMatrixDeltaPrior) {
            setState(edgeMatrixDeltaPrior.getState());
        }
    virtual ~EdgeMatrixDeltaPrior(){}
    const EdgeMatrixDeltaPrior& operator=(const EdgeMatrixDeltaPrior& other){
        this->setState(other.getState());
        return *this;
    }

    void setState(const MultiGraph& edgeMatrix) {
        m_edgeMatrix = edgeMatrix;
        m_state = edgeMatrix;
    }
    void sampleState() override { };
    void samplePriors() override { };

    const double getLogLikelihood() const override { return 0.; }
    const double getLogPrior() const override { return 0.; };

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override ;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override ;
    const double getLogPriorRatioFromGraphMove(const GraphMove& move) const override { return 0.; };
    const double getLogPriorRatioFromLabelMove(const BlockMove& move) const override{ return 0.; }

    void checkSelfConsistency() const override { };
    void checkSelfSafety() const override {
        if (m_edgeMatrix.getSize() == 0)
            throw SafetyError("EdgeMatrixDeltaPrior: unsafe prior since `m_edgeMatrix` is empty.");
    }

    virtual void computationFinished() const override { m_isProcessed = false; }
};

class EdgeMatrixUniformPrior: public EdgeMatrixPrior {
public:
    using EdgeMatrixPrior::EdgeMatrixPrior;
    void sampleState() override;
    const double getLogLikelihood() const override {
        return getLogLikelihood(m_blockPriorPtr->getEffectiveBlockCount(), m_edgeCountPriorPtr->getState());
    }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const override;

private:
    double getLogLikelihoodRatio(size_t blockCountAfter, size_t edgeNumberAfter) const {
        return getLogLikelihood(blockCountAfter, edgeNumberAfter)
        -getLogLikelihood(m_edgeCountPriorPtr->getState(), m_blockPriorPtr->getEffectiveBlockCount());

    }
    double getLogLikelihood(size_t blockCount, size_t edgeCount) const {
        return -logMultisetCoefficient( blockCount*(blockCount+1)/2, edgeCount );
    }
};

// class EdgeMatrixExponentialPrior: public EdgeMatrixPrior {
// public:
//
//     EdgeMatrixExponentialPrior() {}
//     EdgeMatrixExponentialPrior(double edgeCountMean, BlockPrior& blockPrior):
//         EdgeMatrixPrior(), m_edgeCountMean(edgeCountMean){
//         setEdgeCountPrior(*new EdgeCountDeltaPrior(0));
//         setBlockPrior(blockPrior);
//     }
//     EdgeMatrixExponentialPrior(const EdgeMatrixExponentialPrior& other){
//         setEdgeCountPrior(*other.m_edgeCountPriorPtr);
//         setBlockPrior(*other.m_blockPriorPtr);
//     }
//     const EdgeMatrixExponentialPrior& operator=(const EdgeMatrixExponentialPrior& other){
//         setEdgeCountPrior(*other.m_edgeCountPriorPtr);
//         setBlockPrior(*other.m_blockPriorPtr);
//         return *this;
//     }
//     virtual ~EdgeMatrixExponentialPrior(){
//         delete m_edgeCountPriorPtr;
//     }
//     void sampleState() override;
//     double getLogLikelihood() const override {
//         return getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
//     }
//     double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override;
//     double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const override;
//
// private:
//     double m_edgeCountMean;
//     size_t m_edgeCount;
//     double getLikelihoodRatio(size_t blockCountAfter, size_t edgeNumberAfter) const {
//         return getLogLikelihood(m_edgeCountPriorPtr->getState(), m_blockPriorPtr->getBlockCount())
//             - getLogLikelihood(blockCountAfter, edgeNumberAfter);
//     }
//     double getLogLikelihood(size_t blockCount, size_t edgeCount) const {
//         return edgeCount * log(m_edgeCountMean / (m_edgeCountMean + 1))
//              - blockCount * (blockCount + 1) / 2 * log(m_edgeCountMean + 1);
//     }
// };


} // namespace FastMIDyNet

#endif
