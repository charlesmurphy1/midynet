#ifndef FAST_MIDYNET_LABEL_GRAPH_H
#define FAST_MIDYNET_LABEL_GRAPH_H

#include "prior.hpp"
#include "edge_count.h"
#include "block.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"
#include "FastMIDyNet/exceptions.h"
#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/utility/maps.hpp"
#include "FastMIDyNet/generators.h"


namespace FastMIDyNet{

class LabelGraphPrior: public BlockLabeledPrior< MultiGraph >{
    protected:
        EdgeCountPrior* m_edgeCountPriorPtr = nullptr;
        BlockPrior* m_blockPriorPtr = nullptr;
        CounterMap<BlockIndex> m_edgeCounts;
        const MultiGraph* m_graphPtr;

        void _samplePriors() override { m_edgeCountPriorPtr->sample(); m_blockPriorPtr->sample(); }
        const double _getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint() + m_blockPriorPtr->getLogJoint(); }

        void _applyGraphMove(const GraphMove& move) override {
            m_edgeCountPriorPtr->applyGraphMove(move);
            m_blockPriorPtr->applyGraphMove(move);
            applyGraphMoveToState(move);
        }
        void _applyLabelMove(const BlockMove& move) override {
            m_blockPriorPtr->applyLabelMove(move);
            applyLabelMoveToState(move);
        }

        const double _getLogPriorRatioFromGraphMove(const GraphMove& move) const override {
            return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move) + m_blockPriorPtr->getLogJointRatioFromGraphMove(move);
        }
        const double _getLogPriorRatioFromLabelMove(const BlockMove& move) const override {
            return m_blockPriorPtr->getLogJointRatioFromLabelMove(move);
        }

        virtual void applyGraphMoveToState(const GraphMove&);
        virtual void applyLabelMoveToState(const BlockMove&);
        virtual void recomputeConsistentState() ;
        virtual void recomputeStateFromGraph() ;
        CounterMap<BlockIndex> computeEdgeCountsFromState(const MultiGraph& state){
            CounterMap<BlockIndex> edgeCounts;
            for (auto v : state)
                edgeCounts.set(v, state.getDegreeOfIdx(v));
            return edgeCounts;
        }
    public:
        LabelGraphPrior() {}
        LabelGraphPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior){
                setEdgeCountPrior(edgeCountPrior);
                setBlockPrior(blockPrior);
            }
        LabelGraphPrior(const LabelGraphPrior& other){
            setEdgeCountPrior(*other.m_edgeCountPriorPtr);
            setBlockPrior(*other.m_blockPriorPtr);
        }
        const LabelGraphPrior& operator=(const LabelGraphPrior& other){
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
        void setPartition(const std::vector<BlockIndex>&) ;
        void samplePartition() {
            m_blockPriorPtr->sampleState();
            recomputeConsistentState();
        }

        const size_t& getEdgeCount() const { return m_edgeCountPriorPtr->getState(); }
        const CounterMap<BlockIndex>& getEdgeCounts() const { return m_edgeCounts; }


        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
        virtual const double getLogLikelihoodRatioFromLabelMove(const BlockMove&) const = 0;


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
        virtual void checkSelfConsistency() const override;

        void checkSelfSafety()const override{
            if (m_blockPriorPtr == nullptr)
                throw SafetyError("LabelGraphPrior: unsafe prior since `m_blockPriorPtr` is empty.");
            if (m_edgeCountPriorPtr == nullptr)
                throw SafetyError("LabelGraphPrior: unsafe prior since `m_edgeCountPriorPtr` is empty.");
            m_blockPriorPtr->checkSafety();
            m_edgeCountPriorPtr->checkSafety();
        }
};

class LabelGraphDeltaPrior: public LabelGraphPrior{
public:
    MultiGraph m_labelGraph;
    EdgeCountDeltaPrior m_edgeCountDeltaPrior;

public:
    LabelGraphDeltaPrior(){}
    LabelGraphDeltaPrior(const MultiGraph& labelGraph) {
        setState(labelGraph);
        m_edgeCountDeltaPrior.setState(labelGraph.getTotalEdgeNumber());
    }
    LabelGraphDeltaPrior(const MultiGraph& labelGraph, EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior):
        LabelGraphPrior(edgeCountPrior, blockPrior){ setState(labelGraph); }

    LabelGraphDeltaPrior(const LabelGraphDeltaPrior& labelGraphDeltaPrior):
        LabelGraphPrior(labelGraphDeltaPrior) {
            setState(labelGraphDeltaPrior.getState());
        }
    virtual ~LabelGraphDeltaPrior(){}
    const LabelGraphDeltaPrior& operator=(const LabelGraphDeltaPrior& other){
        this->setState(other.getState());
        return *this;
    }

    void setState(const MultiGraph& labelGraph) {
        m_labelGraph = labelGraph;
        m_state = labelGraph;
        m_edgeCounts.clear();
        for (const auto r : m_labelGraph)
            m_edgeCounts.set(r, m_labelGraph.getDegreeOfIdx(r));
        // recomputeState();
    }
    void sampleState() override { };

    const double getLogLikelihood() const override { return 0.; }

    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const override ;
    const double getLogLikelihoodRatioFromLabelMove(const BlockMove& move) const override ;

    void checkSelfConsistency() const override { };
    void checkSelfSafety() const override {
        if (m_labelGraph.getSize() == 0)
            throw SafetyError("LabelGraphDeltaPrior: unsafe prior since `m_labelGraph` is empty.");
    }

    virtual void computationFinished() const override { m_isProcessed = false; }
};

class LabelGraphErdosRenyiPrior: public LabelGraphPrior {
public:
    using LabelGraphPrior::LabelGraphPrior;
    void sampleState() override {
        setState(generateMultiGraphErdosRenyi(m_blockPriorPtr->getBlockCount(), m_edgeCountPriorPtr->getState(), true));
    }
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

// class LabelGraphExponentialPrior: public LabelGraphPrior {
// public:
//
//     LabelGraphExponentialPrior() {}
//     LabelGraphExponentialPrior(double edgeCountMean, BlockPrior& blockPrior):
//         LabelGraphPrior(), m_edgeCountMean(edgeCountMean){
//         setEdgeCountPrior(*new EdgeCountDeltaPrior(0));
//         setBlockPrior(blockPrior);
//     }
//     LabelGraphExponentialPrior(const LabelGraphExponentialPrior& other){
//         setEdgeCountPrior(*other.m_edgeCountPriorPtr);
//         setBlockPrior(*other.m_blockPriorPtr);
//     }
//     const LabelGraphExponentialPrior& operator=(const LabelGraphExponentialPrior& other){
//         setEdgeCountPrior(*other.m_edgeCountPriorPtr);
//         setBlockPrior(*other.m_blockPriorPtr);
//         return *this;
//     }
//     virtual ~LabelGraphExponentialPrior(){
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
