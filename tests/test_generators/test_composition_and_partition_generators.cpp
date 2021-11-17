#include "gtest/gtest.h"

#include "FastMIDyNet/generators.h"


static const size_t N = 10;
static const size_t K = 4;
static FastMIDyNet::RNG RNG;


TEST(TestCompositionGenerator, generateComposition_givenNAndK_generatedListOfKnumbers) {
    for (int i=0; i<10; i++) {
        std::list<size_t> composition = FastMIDyNet::sampleRandomComposition(N, K, RNG);
        EXPECT_EQ(composition.size(), K);
    }
}


TEST(TestCompositionGenerator, generateComposition_givenNAndK_compositionSumsToN) {
    for (int i=0; i<10; i++) {
        std::list<size_t> composition = FastMIDyNet::sampleRandomComposition(N, K, RNG);
        size_t sum=0;
        for (size_t element: composition)
            sum += element;
        EXPECT_EQ(sum, N);
    }
}


TEST(TestCompositionGenerator, generateComposition_givenNAndK_compositionHasNoZeros) {
    for (int i=0; i<10; i++) {
        std::list<size_t> composition = FastMIDyNet::sampleRandomComposition(N, K, RNG);
        for (size_t element: composition) {
            EXPECT_NE(element, 0);
            EXPECT_NE(element, N);
        }
    }
}


TEST(TestWeakCompositionGenerator, generateWeakComposition_givenNAndK_generatedListOfKNumbers) {
    for (int i=0; i<10; i++) {
        std::list<size_t> weakComposition = FastMIDyNet::sampleRandomWeakComposition(N, K, RNG);
        EXPECT_EQ(weakComposition.size(), K);
    }
}


TEST(TestWeakCompositionGenerator, generateWeakComposition_givenNAndK_compositionSumsToN) {
    for (int i=0; i<10; i++) {
        std::list<size_t> weakComposition = FastMIDyNet::sampleRandomWeakComposition(N, K, RNG);
        size_t sum=0;
        for (size_t element: weakComposition)
            sum += element;
        EXPECT_EQ(sum, N);
    }
}


TEST(TestRestrictedPartitionGenerator, generateRestrictedPartition_givenNAndK_returnListOfKNumbers) {
    for (int i=0; i<10; i++) {
        std::list<size_t> partition = FastMIDyNet::sampleRandomRestrictedPartition(N, K, RNG);
        EXPECT_EQ(partition.size(), K);
    }
}


TEST(TestRestrictedPartitionGenerator, generateRestrictedPartition_givenNAndK_partitionSumsToN) {
    for (int i=0; i<10; i++) {
        std::list<size_t> partition = FastMIDyNet::sampleRandomRestrictedPartition(N, K, RNG);
        size_t sum=0;
        for (size_t element: partition)
            sum += element;
        EXPECT_EQ(sum, N);
    }
}
