#ifndef FAST_MIDYNET_COLLECTOR_HPP
#define FAST_MIDYNET_COLLECTOR_HPP

#include <vector>
#include <fstream>

#include "callback.h"
#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "FastMIDyNet/utility/distance.h"
#include "BaseGraph/fileio.h"

namespace FastMIDyNet{

template<typename MCMCType>
class Collector: public CallBack<MCMCType>{
public:
    virtual void onBegin() { clear(); }
    virtual void collect() = 0;
    virtual void clear() = 0;
};

template<typename MCMCType>
class SweepCollector: public Collector<MCMCType>{
public:
    virtual void onSweepEnd() { Collector<MCMCType>::collect(); }
};

template<typename MCMCType>
class StepCollector: public Collector<MCMCType>{
public:
    virtual void onStepEnd() { Collector<MCMCType>::collect(); }
};

class CollectGraphOnSweep: public SweepCollector<RandomGraphMCMC>{
private:
    std::vector<MultiGraph> m_collectedGraphs;
public:
    void collect() override { m_collectedGraphs.push_back( m_mcmcPtr->getGraph() ); }
    void clear() override { m_collectedGraphs.clear(); }
    const std::vector<MultiGraph>& getGraphs() const { return m_collectedGraphs; }
};

class CollectEdgeMultiplicityOnSweep: public SweepCollector<RandomGraphMCMC>{
private:
    CounterMap<BaseGraph::Edge> m_observedEdges;
    CounterMap<std::pair<BaseGraph::Edge, size_t>> m_observedEdgesCount;
    CounterMap<BaseGraph::Edge> m_observedEdgesMaxCount;
    size_t m_totalCount;
public:
    void setUp(RandomGraphMCMC* mcmcPtr) {
        Collector::setUp(mcmcPtr);
    }
    void collect() override ;
    void clear() override { m_observedEdges.clear(); m_observedEdgesCount.clear(); m_observedEdgesMaxCount.clear();}
    const double getMarginalEntropy() ;
    const double getLogPosteriorEstimate(const MultiGraph&) ;
    const double getLogPosteriorEstimate() { return getLogPosteriorEstimate(m_mcmcPtr->getGraph()); }
    size_t getTotalCount() const { return m_totalCount; }
    size_t getEdgeObservationCount(BaseGraph::Edge edge) const { return m_observedEdges[edge]; }
    const double getEdgeCountProb(BaseGraph::Edge edge, size_t count) const ;
    const std::map<BaseGraph::Edge, std::vector<double>> getEdgeProbs() ;

};

class CollectPartitionOnSweep: public SweepCollector<StochasticBlockModelMCMC>{
private:
    std::vector<std::vector<BlockIndex>> m_partitions;
public:
    void setUp(StochasticBlockModelMCMC* mcmcPtr) { Collector::setUp(mcmcPtr); }
    void collect() override { m_partitions.push_back(m_mcmcPtr->getVertexLabels()); }
    void clear() override { m_partitions.clear(); }
    const std::vector<BlockSequence>& getPartitions() const { return m_partitions; }
};

class WriteGraphToFileOnSweep: public SweepCollector<RandomGraphMCMC>{
private:
    std::string m_filename;
    std::string m_ext;
public:
    WriteGraphToFileOnSweep(std::string filename, std::string ext=".b"):
    m_filename(filename), m_ext(ext) {}
    void collect() override ;
    void clear() override { };
};

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

class CollectGraphDistance: public Collector<RandomGraphMCMC>{
private:
    MultiGraph m_originalGraph;
    const GraphDistance& m_distance;
    std::vector<double> m_collectedDistances;
public:
    CollectGraphDistance(const GraphDistance& distance): m_distance(distance){}
    const std::vector<double>& getCollectedDistances() { return m_collectedDistances; }
    void onSweepBegin() { m_originalGraph = m_mcmcPtr->getGraph(); }
    void collect() {
        m_collectedDistances.push_back(
            m_distance.compute( m_originalGraph, m_mcmcPtr->getGraph() )
        );
    }
    void clear() { m_collectedDistances.clear(); };
    void onStepEnd() { collect(); }
};

}

#endif
