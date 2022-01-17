#include "gtest/gtest.h"

#include "fixtures.hpp"
#include "FastMIDyNet/mcmc/graph_mcmc.h"
#include "FastMIDyNet/mcmc/callbacks/callback.h"
#include "FastMIDyNet/mcmc/callbacks/collector.h"
namespace FastMIDyNet{

#define CALLBACK_TESTS(TEST_CALL_BACK, TESTED_CALL_CLASS)\
    class TEST_CALL_BACK: public::testing::Test{\
    public:\
        DummyRandomGraph g = DummyRandomGraph();\
        StochasticBlockGraphMCMC mcmc = StochasticBlockGraphMCMC(g.randomGraph, g.edgeProposer,g.blockProposer);\
        TESTED_CALL_CLASS callback = TESTED_CALL_CLASS();\
        void SetUp(){\
            seedWithTime();\
            mcmc.addCallBack(callback);\
            mcmc.setUp();\
        }\
        void TearDown(){\
            mcmc.tearDown();\
            mcmc.popCallBack();\
        }\
    };\
    TEST_F(TEST_CALL_BACK, tearDown_raiseNoExceptionOrSegFault){\
        callback.tearDown();\
    }\
    TEST_F(TEST_CALL_BACK, onBegin_raiseNoExceptionOrSegFault){\
        callback.onBegin();\
    }\
    TEST_F(TEST_CALL_BACK, onEnd_raiseNoExceptionOrSegFault){\
        callback.onEnd();\
    }\
    TEST_F(TEST_CALL_BACK, onStepBegin_raiseNoExceptionOrSegFault){\
        callback.onStepBegin();\
    }\
    TEST_F(TEST_CALL_BACK, onStepEnd_raiseNoExceptionOrSegFault){\
        callback.onStepEnd();\
    }\
    TEST_F(TEST_CALL_BACK, onSweepBegin_raiseNoExceptionOrSegFault){\
        callback.onSweepBegin();\
    }\
    TEST_F(TEST_CALL_BACK, onSweepEnd_raiseNoExceptionOrSegFault){\
        callback.onSweepEnd();\
    }

CALLBACK_TESTS(TestCallBackBaseClass, CallBack);

CALLBACK_TESTS(TestCollectGraphOnSweep, CollectGraphOnSweep);
TEST_F(TestCollectGraphOnSweep, onSweepEnd_collectGraph_expectGraphVectorNotEmpty){
    callback.onSweepEnd();
    EXPECT_EQ(callback.getGraphs().size(), 1);
}

CALLBACK_TESTS(TestCollectEdgeMultiplicityOnSweep, CollectEdgeMultiplicityOnSweep);

CALLBACK_TESTS(TestCollectPartitionOnSweep, CollectPartitionOnSweep);

// CALLBACK_TESTS(TestWriteGraphToFileOnSweep, WriteGraphToFileOnSweep);

CALLBACK_TESTS(TestCollectLikelihoodOnSweep, CollectLikelihoodOnSweep);
TEST_F(TestCollectLikelihoodOnSweep, onSweepEnd_collectLikelihood_expectLikelihoodVectorNotEmpty){
    callback.onSweepEnd();
    EXPECT_EQ(callback.getLogLikelihoods().size(), 1);
}

CALLBACK_TESTS(TestCollectPriorOnSweep, CollectPriorOnSweep);
TEST_F(TestCollectPriorOnSweep, onSweepEnd_collectPrior_expectPriorVectorNotEmpty){
    callback.onSweepEnd();
    EXPECT_EQ(callback.getLogPriors().size(), 1);
}

CALLBACK_TESTS(TestCollectJointOnSweep, CollectJointOnSweep);
TEST_F(TestCollectJointOnSweep, onSweepEnd_collectJoint_expectJointVectorNotEmpty){
    callback.onSweepEnd();
    EXPECT_EQ(callback.getLogJoints().size(), 1);
}

}
