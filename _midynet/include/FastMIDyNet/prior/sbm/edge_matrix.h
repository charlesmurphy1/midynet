#ifndef FAST_MIDYNET_EDGE_MATRIX_H
#define FAST_MIDYNET_EDGE_MATRIX_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_count.h"
#include "FastMIDyNet/prior/sbm/block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"


namespace FastMIDyNet{

class EdgeMatrixPrior: public Prior< EdgeMatrix >{
    protected:
        EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
        BlockPrior* m_blockPriorPtr = nullptr;
        std::vector<size_t> m_edgeCountsInBlocks;
        const MultiGraph* m_graph;

        void createBlock();
        void destroyBlock(const BlockIndex&);
        void moveEdgeCountsInBlocks(const BlockMove& move);
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
        void setBlockPrior(BlockPrior& blockPrior) { m_blockPriorPtr = &blockPrior;  m_blockPriorPtr->isRoot(false);}

        void setGraph(const MultiGraph& graph);
        const MultiGraph& getGraph() { return *m_graph; }
        void setState(const EdgeMatrix&) override;

        const size_t& getEdgeCount() const { return m_edgeCountPriorPtr->getState(); }
        const std::vector<size_t>& getEdgeCountsInBlocks() const { return m_edgeCountsInBlocks; }


        void samplePriors() override { m_edgeCountPriorPtr->sample(); m_blockPriorPtr->sample(); }

        double getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint() + m_blockPriorPtr->getLogJoint(); }

        virtual double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
        virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const = 0;

        double getLogPriorRatioFromGraphMove(const GraphMove& move) const { return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move) + m_blockPriorPtr->getLogJointRatioFromGraphMove(move); }
        double getLogPriorRatioFromBlockMove(const BlockMove& move) const { return m_edgeCountPriorPtr->getLogJointRatioFromBlockMove(move) + m_blockPriorPtr->getLogJointRatioFromBlockMove(move); }

        double getLogJointRatioFromGraphMove(const GraphMove& move) const {
            return processRecursiveConstFunction<double>( [&]() { return getLogLikelihoodRatioFromGraphMove(move) + getLogPriorRatioFromGraphMove(move); }, 0);
        }

        double getLogJointRatioFromBlockMove(const BlockMove& move) const  {
            return processRecursiveConstFunction<double>( [&]() { return getLogLikelihoodRatioFromBlockMove(move) + getLogPriorRatioFromBlockMove(move); }, 0);
        }

        void applyGraphMoveToState(const GraphMove&);
        void applyBlockMoveToState(const BlockMove&);
        void applyGraphMove(const GraphMove& move){
            processRecursiveFunction( [&]() { m_edgeCountPriorPtr->applyGraphMove(move); m_blockPriorPtr->applyGraphMove(move); applyGraphMoveToState(move); } );
            #if DEBUG
            checkSelfConsistency();
            #endif
        }
        void applyBlockMove(const BlockMove& move) {
            processRecursiveFunction( [&]() { m_edgeCountPriorPtr->applyBlockMove(move); m_blockPriorPtr->applyBlockMove(move); applyBlockMoveToState(move); } );
            #if DEBUG
            checkSelfConsistency();
            #endif
        }

        void computationFinished() const override {
            m_isProcessed = false;
            m_blockPriorPtr->computationFinished();
            m_edgeCountPriorPtr->computationFinished();
        }
        void checkSelfConsistencywithGraph() const;
        void checkSelfConsistency() const override;

        void checkSafety()const override{
            if (m_blockPriorPtr == nullptr)
                throw SafetyError("EdgeMatrixPrior: unsafe prior since `m_blockPriorPtr` is empty.");
            if (m_edgeCountPriorPtr == nullptr)
                throw SafetyError("EdgeMatrixPrior: unsafe prior since `m_edgeCountPriorPtr` is empty.");
        }
};

class EdgeMatrixUniformPrior: public EdgeMatrixPrior {
public:
    using EdgeMatrixPrior::EdgeMatrixPrior;
    void sampleState() override;
    double getLogLikelihood() const override {
        return getLogLikelihood(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState());
    }
    double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const override;
    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const override;

private:
    double getLikelihoodRatio(size_t blockCountAfter, size_t edgeNumberAfter) const {
        return getLogLikelihood(m_edgeCountPriorPtr->getState(), m_blockPriorPtr->getBlockCount())
            - getLogLikelihood(blockCountAfter, edgeNumberAfter);
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
//     double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const override;
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
