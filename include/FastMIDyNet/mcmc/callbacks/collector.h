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

class DynamicsCollector: public Collector{

protected:
    DynamicsMCMC* m_dynamicsMCMCPtr;
public:
    virtual void setUp(DynamicsMCMC* mcmcPtr) {
        m_mcmcPtr = mcmcPtr;
        m_dynamicsMCMCPtr = m_dynamicsMCMCPtr;
    }
};

class CollectGraphOnSweep: public SweepCollector, public DynamicsCollector{
private:
    std::vector<MultiGraph> m_collectedGraphs;
public:
    void collect() { m_collectedGraphs.push_back( m_dynamicsMCMCPtr->getGraph() ); }
    void clear() { m_collectedGraphs.clear(); }
};

class CollectEdgeMultiplicityOnSweep: public SweepCollector, public DynamicsCollector{
private:
    MultiGraph m_edgeMultiplicity;
public:
    void setUp(DynamicsMCMC* mcmcPtr) {
        DynamicsCollector::setUp(mcmcPtr);
        m_edgeMultiplicity = MultiGraph(m_dynamicsMCMCPtr->getSize());
    }
    void collect() ;
    void clear() { m_edgeMultiplicity.clearEdges(); }
};

class WriteGraphToFileOnSweep: public SweepCollector, public DynamicsCollector{
private:
    std::string m_filename;
    std::string m_ext;
public:
    WriteGraphToFileOnSweep(std::string filename, std::string ext=".b"):
    m_filename(filename), m_ext(ext) {}
    void collect() ;
    void clear() { };
};

class CollectLikelihoodOnSweep: public SweepCollector, public DynamicsCollector{
private:
    std::vector<double> m_collectedLikelihoods;
public:
    void collect() { m_collectedLikelihoods.push_back( m_dynamicsMCMCPtr->getLogLikelihood() ); }
    void clear() { m_collectedLikelihoods.clear(); }
};

class CollectPriorOnSweep: public SweepCollector, public DynamicsCollector{
private:
    std::vector<double> m_collectedPriors;
public:
    void collect() { m_collectedPriors.push_back( m_dynamicsMCMCPtr->getLogPrior() ); }
    void clear() { m_collectedPriors.clear(); }
};

class CollectJointOnSweep: public SweepCollector, public DynamicsCollector{
private:
    std::vector<double> m_collectedJoints;
public:
    void collect() { m_collectedJoints.push_back( m_dynamicsMCMCPtr->getLogJoint() ); }
    void clear() { m_collectedJoints.clear(); }
};

}

#endif
