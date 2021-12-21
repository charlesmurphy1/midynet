#ifndef FAST_MIDYNET_DEGREE_H
#define FAST_MIDYNET_DEGREE_H

#include "FastMIDyNet/prior/prior.hpp"
#include "FastMIDyNet/prior/sbm/edge_matrix.h"
#include "FastMIDyNet/prior/sbm/degree_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/utility/maps.hpp"


namespace FastMIDyNet{

class DegreePrior: public Prior< DegreeSequence >{
    public:
        using Prior::Prior;
        typedef std::vector<CounterMap<size_t>> DegreeCountsMap;
        DegreePrior(BlockPrior& blockPrior, EdgeMatrixPrior& edgeMatrixPrior):
            m_blockPrior(blockPrior), m_edgeMatrixPrior(edgeMatrixPrior) {
                m_blockPrior.isRoot(false);
                m_edgeMatrixPrior.isRoot(false);
            }

        void setGraph(const MultiGraph&);
        const MultiGraph& getGraph() const { return *m_graph; }
        void setState(const DegreeSequence&) override;

        const size_t& getSize() const { return m_blockPrior.getSize(); }
        const BlockIndex& getBlockOfIdx(BaseGraph::VertexIndex idx) const { return m_blockPrior.getBlockOfIdx(idx); }
        const BlockIndex& getDegreeOfIdx(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
        const size_t& getEdgeCount() const { return m_edgeMatrixPrior.getEdgeCount(); }
        const size_t& getBlockCount() const { return m_blockPrior.getBlockCount(); }
        const std::vector<size_t>& getEdgeCountsInBlocks() const { return m_edgeMatrixPrior.getEdgeCountsInBlocks(); }
        const EdgeMatrix& getEdgeMatrix() const { return m_edgeMatrixPrior.getState(); }
        const BlockSequence& getBlocks() const { return m_blockPrior.getState(); }
        const std::vector<size_t>& getVertexCountsInBlocks() const { return m_blockPrior.getVertexCountsInBlocks(); }
        static DegreeCountsMap computeDegreeCountsInBlocks(const DegreeSequence& degreeSeq, const BlockSequence& blockSeq);
        virtual const DegreeCountsMap& getDegreeCountsInBlocks() const { return m_degreeCountsInBlocks; }

        BlockPrior& getBlockPrior() const { return m_blockPrior; }
        EdgeMatrixPrior& getEdgeMatrixPrior() const { return m_edgeMatrixPrior; }

        void samplePriors() override { m_blockPrior.sample(); m_edgeMatrixPrior.sample(); }

        double getLogPrior() override { return m_blockPrior.getLogJoint() + m_edgeMatrixPrior.getLogJoint(); }

        virtual double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const = 0;
        virtual double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const = 0;

        double getLogPriorRatioFromGraphMove(const GraphMove& move) { return m_blockPrior.getLogJointRatioFromGraphMove(move) + m_edgeMatrixPrior.getLogJointRatioFromGraphMove(move); }
        double getLogPriorRatioFromBlockMove(const BlockMove& move) { return m_blockPrior.getLogJointRatioFromBlockMove(move) + m_edgeMatrixPrior.getLogJointRatioFromBlockMove(move); }

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

        void computationFinished() override {
            m_isProcessed = false;
            m_blockPrior.computationFinished();
            m_edgeMatrixPrior.computationFinished();
        }
        static void checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence&, size_t);
        static void checkDegreeSequenceConsistencyWithDegreeCountsInBlocks(const DegreeSequence&, const BlockSequence&, const std::vector<CounterMap<size_t>>&);
        void checkSelfConsistency() const override;

    protected:
        EdgeMatrixPrior& m_edgeMatrixPrior;
        BlockPrior& m_blockPrior;
        DegreeCountsMap m_degreeCountsInBlocks;
        const MultiGraph* m_graph;

        void createBlock();
        void destroyBlock(const BlockIndex&);
};

class DegreeUniformPrior: public DegreePrior{
public:
    using DegreePrior::DegreePrior;
    void sampleState();

    double getLogLikelihood() const;
    double getLogLikelihoodRatioFromGraphMove(const GraphMove&) const;
    double getLogLikelihoodRatioFromBlockMove(const BlockMove&) const;
};

class DegreeHyperPrior: public DegreePrior{

};


} // FastMIDyNet

#endif
