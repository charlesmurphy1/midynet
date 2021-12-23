#ifndef FAST_MIDYNET_COLLECTOR_HPP
#define FAST_MIDYNET_COLLECTOR_HPP

#include <vector>
#include <fstream>

#include "callback.h"
#include "FastMIDyNet/mcmc/dynamics_mcmc.h"
#include "BaseGraph/fileio.h"

namespace FastMIDyNet{

class Collector: public CallBack{
public:
    virtual void onBegin() { clear(); }
    virtual void collect() = 0;
    virtual void clear() = 0;
};

class SweepCollector: public Collector{
public:
    virtual void onSweepEnd() { collect(); }
};

class StepCollector: public Collector{
public:
    virtual void onStepEnd() { collect(); }
};

class CollectGraphOnSweep: public SweepCollector{
private:
    std::vector<MultiGraph> m_collectedGraphs;
public:
    void collect() { m_collectedGraphs.push_back( m_mcmcPtr->getGraph() ); }
    void clear() { m_collectedGraphs.clear(); }
    const std::vector<MultiGraph>& getGraphs() const { return m_collectedGraphs; }
};

class CollectEdgeMultiplicityOnSweep: public SweepCollector{
private:
    MultiGraph m_edgeMultiplicity;
public:
    void setUp(MCMC* mcmcPtr) {
        Collector::setUp(mcmcPtr);
        m_edgeMultiplicity = MultiGraph(m_mcmcPtr->getSize());
    }
    void collect() ;
    void clear() { m_edgeMultiplicity.clearEdges(); }
    const MultiGraph& getEdgeMultiplicity() const { return m_edgeMultiplicity; }
};

class WriteGraphToFileOnSweep: public SweepCollector{
private:
    std::string m_filename;
    std::string m_ext;
public:
    WriteGraphToFileOnSweep(std::string filename, std::string ext=".b"):
    m_filename(filename), m_ext(ext) {}
    void collect() ;
    void clear() { };
};

class CollectLikelihoodOnSweep: public SweepCollector{
private:
    std::vector<double> m_collectedLikelihoods;
public:
    void collect() { m_collectedLikelihoods.push_back( m_mcmcPtr->getLogLikelihood() ); }
    void clear() { m_collectedLikelihoods.clear(); }
    const std::vector<double>& getLogLikelihoods() const { return m_collectedLikelihoods; }
};

class CollectPriorOnSweep: public SweepCollector{
private:
    std::vector<double> m_collectedPriors;
public:
    void collect() { m_collectedPriors.push_back( m_mcmcPtr->getLogPrior() ); }
    void clear() { m_collectedPriors.clear(); }
    const std::vector<double>& getLogPriors() const { return m_collectedPriors; }
};

class CollectJointOnSweep: public SweepCollector{
private:
    std::vector<double> m_collectedJoints;
public:
    void collect() { m_collectedJoints.push_back( m_mcmcPtr->getLogJoint() ); }
    void clear() { m_collectedJoints.clear(); }
    const std::vector<double>& getLogJoints() const { return m_collectedJoints; }

};

}

#endif
