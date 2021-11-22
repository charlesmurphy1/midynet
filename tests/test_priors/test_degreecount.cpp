#include "gtest/gtest.h"

#include "FastMIDyNet/prior/dcsbm/degree_count.h"
#include "FastMIDyNet/prior/dcsbm/edge_count.h"
#include "FastMIDyNet/proposer/movetypes.h"
#include "FastMIDyNet/types.h"


class DummyDegreeCountPrior: public FastMIDyNet::DegreeCountPrior {
    public:
        DummyDegreeCountPrior(size_t graphSize, FastMIDyNet::EdgeCountPrior& edgeCountPrior):
        DegreeCountPrior(graphSize, edgeCountPrior){}
        void sampleState() { }

        double getLogLikelihood(const FastMIDyNet::DegreeSequence& state) const { return 0; }
        double getLogLikelihood()const { return getLogLikelihood(m_state); }
        double getLogLikelihoodRatio(const FastMIDyNet::GraphMove& move) const { return 0; }

        void checkSelfConsistency() const {}
};

class TestDegreeCountPrior: public ::testing::Test {
    public:
        double poissonMean = 10;
        size_t size = 10;
        FastMIDyNet::EdgeCountPoissonPrior edgeCountPrior = FastMIDyNet::EdgeCountPoissonPrior(5);
        DummyDegreeCountPrior degreeCountPrior = DummyDegreeCountPrior(size, edgeCountPrior);
};


// TEST_F(TestDegreeCountPrior, getLogPrior_anyState_returnHyperPriorLogLikelihood) {
//     EXPECT_EQ(edgeCountPrior.getLogLikelihood(poissonMean), degreeCountPrior.getLogPrior());
// }
