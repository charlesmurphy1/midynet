#ifndef FAST_MIDYNET_COLLECTOR_HPP
#define FAST_MIDYNET_COLLECTOR_HPP

#include <vector>
#include <fstream>

#include "callback.hpp"
#include "FastMIDyNet/mcmc/community.hpp"
#include "FastMIDyNet/mcmc/reconstruction.hpp"
#include "FastMIDyNet/utility/distance.h"
#include "BaseGraph/fileio.h"

namespace FastMIDyNet{

template<typename MCMCType>
class Collector: public CallBack<MCMCType>{
public:
    virtual void onBegin() override { CallBack<MCMCType>::clear(); }
    virtual void collect() = 0;
};

using BlockCollector = Collector<BlockLabelMCMC>;
using GraphReconstructionCollector = Collector<GraphReconstructionMCMC<RandomGraph>>;
using BlockLabeledGraphReconstructionCollector = Collector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>;

template<typename MCMCType>
class SweepCollector: public Collector<MCMCType>{
public:
    void onSweepEnd() override { this->collect(); }
};

using BlockSweepCollector = SweepCollector<BlockLabelMCMC>;
using GraphReconstructionSweepCollector = SweepCollector<GraphReconstructionMCMC<RandomGraph>>;
using BlockLabeledGraphReconstructionSweepCollector = SweepCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>;

template<typename MCMCType>
class StepCollector: public Collector<MCMCType>{
public:
    void onStepEnd() override { this->collect(); }
};

using BlockStepCollector = StepCollector<BlockLabelMCMC>;
using GraphReconstructionStepCollector = StepCollector<GraphReconstructionMCMC<RandomGraph>>;
using BlockLabeledGraphReconstructionStepCollector = StepCollector<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>;

template<typename GraphMCMC>
class CollectGraphOnSweep: public SweepCollector<GraphMCMC>{
private:
    std::vector<MultiGraph> m_collectedGraphs;
public:
    using BaseClass = SweepCollector<GraphMCMC>;
    void collect() override { m_collectedGraphs.push_back( BaseClass::m_mcmcPtr->getGraph() ); }
    void clear() override { m_collectedGraphs.clear(); }
    const std::vector<MultiGraph>& getGraphs() const { return m_collectedGraphs; }
};

using CollectBlockLabeledGraphOnSweep = CollectGraphOnSweep<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>;

template<typename GraphMCMC>
class CollectEdgeMultiplicityOnSweep: public SweepCollector<GraphMCMC>{
private:
    CounterMap<BaseGraph::Edge> m_observedEdges;
    CounterMap<std::pair<BaseGraph::Edge, size_t>> m_observedEdgesCount;
    CounterMap<BaseGraph::Edge> m_observedEdgesMaxCount;
    size_t m_totalCount;
public:
    using BaseClass = SweepCollector<GraphMCMC>;
    void collect() override ;
    void clear() override { m_observedEdges.clear(); m_observedEdgesCount.clear(); m_observedEdgesMaxCount.clear();}
    const double getMarginalEntropy() ;
    const MultiGraph& getCurrentGraph() { return BaseClass::m_mcmcPtr->getGraph(); }
    const double getLogPosteriorEstimate(const MultiGraph&) ;
    const double getLogPosteriorEstimate() { return getLogPosteriorEstimate(BaseClass::m_mcmcPtr->getGraph()); }
    size_t getTotalCount() const { return m_totalCount; }
    size_t getEdgeObservationCount(BaseGraph::Edge edge) const { return m_observedEdges[edge]; }
    const double getEdgeCountProb(BaseGraph::Edge edge, size_t count) const ;
    const std::map<BaseGraph::Edge, std::vector<double>> getEdgeProbs() ;

};

using CollectBlockLabeledEdgeMultiplicityOnSweep = CollectEdgeMultiplicityOnSweep<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>;

template<typename GraphMCMC>
void CollectEdgeMultiplicityOnSweep<GraphMCMC>::collect(){
    ++m_totalCount;
    const MultiGraph& graph = getCurrentGraph();

    for ( auto vertex : graph){
        for (auto neighbor : graph.getNeighboursOfIdx(vertex)){
            if (vertex <= neighbor.vertexIndex){
                auto edge = getOrderedPair<BaseGraph::VertexIndex>({vertex, neighbor.vertexIndex});
                m_observedEdges.increment(edge);
                m_observedEdgesCount.increment({edge, neighbor.label});
                if (neighbor.label > m_observedEdgesMaxCount[edge])
                    m_observedEdgesMaxCount.set(edge, neighbor.label);
            }
        }
    }
}


template<typename GraphMCMC>
const double CollectEdgeMultiplicityOnSweep<GraphMCMC>::getEdgeCountProb(BaseGraph::Edge edge, size_t count) const {
    if (count == 0)
        return 1.0 - ((double)m_observedEdges.get(edge)) / ((double)m_totalCount);
    else
        return ((double)m_observedEdgesCount.get({edge, count})) / ((double)m_totalCount);
}

template<typename GraphMCMC>
const double CollectEdgeMultiplicityOnSweep<GraphMCMC>::getMarginalEntropy() {
    double marginalEntropy = 0;
    for (auto edge : m_observedEdges){
        for (size_t count = 0; count <= m_observedEdgesMaxCount[edge.first]; ++count){
            double p = getEdgeCountProb(edge.first, count);
            if (p > 0)
                marginalEntropy -= p * log(p);
        }
    }
    return marginalEntropy;
}

template<typename GraphMCMC>
const double CollectEdgeMultiplicityOnSweep<GraphMCMC>::getLogPosteriorEstimate(const MultiGraph& graph) {
    double logPosterior = 0;
    for (auto edge : m_observedEdges)
        logPosterior += log(getEdgeCountProb(edge.first, graph.getEdgeMultiplicityIdx(edge.first)));
    return logPosterior;
}

template<typename GraphMCMC>
const std::map<BaseGraph::Edge, std::vector<double>> CollectEdgeMultiplicityOnSweep<GraphMCMC>::getEdgeProbs() {
    std::map<BaseGraph::Edge, std::vector<double>> edgeProbs;

    for (auto edge : m_observedEdges){
        edgeProbs.insert({edge.first, {}});
        for (size_t count = 0; count <= m_observedEdgesMaxCount[edge.first]; ++count){
            double p = getEdgeCountProb(edge.first, count);
            edgeProbs[edge.first].push_back(p);
        }
    }
    return edgeProbs;
}

template<typename GraphMCMC>
class CollectPartitionOnSweep: public SweepCollector<GraphMCMC>{
private:
    std::vector<std::vector<BlockIndex>> m_partitions;
public:
    using BaseClass = SweepCollector<GraphMCMC>;
    void collect() override { m_partitions.push_back(BaseClass::m_mcmcPtr->getGraphPrior().getVertexLabels()); }
    void clear() override { m_partitions.clear(); }
    const std::vector<BlockSequence>& getPartitions() const { return m_partitions; }
};

using CollectPartitionOnSweepForReconstruction = CollectPartitionOnSweep<GraphReconstructionMCMC<VertexLabeledRandomGraph<BlockIndex>>>;
using CollectPartitionOnSweepForCommunity = CollectPartitionOnSweep<VertexLabelMCMC<BlockIndex>>;

// template<typename GraphMCMC>
// class WriteGraphToFileOnSweep: public SweepCollector<GraphMCMC>{
// private:
//     std::string m_filename;
//     std::string m_ext;
// public:
//     WriteGraphToFileOnSweep(std::string filename, std::string ext=".b"):
//     m_filename(filename), m_ext(ext) {}
//     void collect() override ;
//     void clear() override { };
// };

// template<typename GraphMCMC>
// void WriteGraphToFileOnSweep<GraphMCMC>::collect(){
//     std::ofstream file;
//     file.open(m_filename + "_" + std::to_string(SweepCollector<GraphMCMC>::m_mcmcPtr->getNumSweeps()) + m_ext);
//
//     // BaseGraph::writeEdgeListIdxInBinaryFile(m_mcmcPtr->getGraph(), file);
//     // BaseGraph::writeEdgeListInBinaryFile(m_mcmcPtr->getGraph(), file);
//
//     file.close();
// }

class CollectLikelihoodOnSweep: public SweepCollector<MCMC>{
private:
    std::vector<double> m_collectedLikelihoods;
public:
    void collect() override { m_collectedLikelihoods.push_back( m_mcmcPtr->getLogLikelihood() ); }
    void clear() override { m_collectedLikelihoods.clear(); }
    const std::vector<double>& getLogLikelihoods() const { return m_collectedLikelihoods; }
};

class CollectPriorOnSweep: public SweepCollector<MCMC>{
private:
    std::vector<double> m_collectedPriors;
public:
    void collect() override { m_collectedPriors.push_back( m_mcmcPtr->getLogPrior() ); }
    void clear() override { m_collectedPriors.clear(); }
    const std::vector<double>& getLogPriors() const { return m_collectedPriors; }
};

class CollectJointOnSweep: public SweepCollector<MCMC>{
private:
    std::vector<double> m_collectedJoints;
public:
    void collect() override { m_collectedJoints.push_back( m_mcmcPtr->getLogJoint() ); }
    void clear() override { m_collectedJoints.clear(); }
    const std::vector<double>& getLogJoints() const { return m_collectedJoints; }

};

// class CollectGraphDistance: public Collector<GraphMCMC>{
// private:
//     MultiGraph m_originalGraph;
//     const GraphDistance& m_distance;
//     std::vector<double> m_collectedDistances;
// public:
//     CollectGraphDistance(const GraphDistance& distance): m_distance(distance){}
//     const std::vector<double>& getCollectedDistances() { return m_collectedDistances; }
//     void onSweepBegin() { m_originalGraph = m_mcmcPtr->getGraph(); }
//     void collect() {
//         m_collectedDistances.push_back(
//             m_distance.compute( m_originalGraph, m_mcmcPtr->getGraph() )
//         );
//     }
//     void clear() { m_collectedDistances.clear(); };
//     void onStepEnd() { collect(); }
// };

}

#endif
