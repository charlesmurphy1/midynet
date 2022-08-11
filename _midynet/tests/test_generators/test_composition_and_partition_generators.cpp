#include "gtest/gtest.h"

#include "FastMIDyNet/utility/functions.h"
#include "FastMIDyNet/generators.h"


static const size_t N = 10;
static const size_t K = 4;

namespace FastMIDyNet{

TEST(TestCompositionGenerator, generateComposition_givenNAndK_generatedListOfKnumbers) {
    for (int i=0; i<10; i++) {
        std::list<size_t> composition = sampleRandomComposition(N, K);
        EXPECT_EQ(composition.size(), K);
    }
}


TEST(TestCompositionGenerator, generateComposition_givenNAndK_compositionSumsToN) {
    for (int i=0; i<10; i++) {
        std::list<size_t> composition = sampleRandomComposition(N, K);
        size_t sum=0;
        for (size_t element: composition)
            sum += element;
        EXPECT_EQ(sum, N);
    }
}


TEST(TestCompositionGenerator, generateComposition_givenNAndK_compositionHasNoZeros) {
    for (int i=0; i<10; i++) {
        std::list<size_t> composition = sampleRandomComposition(N, K);
        for (size_t element: composition) {
            EXPECT_NE(element, 0);
            EXPECT_NE(element, N);
        }
    }
}


TEST(TestWeakCompositionGenerator, generateWeakComposition_givenNAndK_generatedListOfKNumbers) {
    for (int i=0; i<10; i++) {
        std::list<size_t> weakComposition = sampleRandomWeakComposition(N, K);
        EXPECT_EQ(weakComposition.size(), K);
    }
}


TEST(TestWeakCompositionGenerator, generateWeakComposition_givenNAndK_compositionSumsToN) {
    for (int i=0; i<10; i++) {
        std::list<size_t> weakComposition = sampleRandomWeakComposition(N, K);
        size_t sum=0;
        for (size_t element: weakComposition)
            sum += element;
        EXPECT_EQ(sum, N);
    }
}


TEST(TestRestrictedPartitionGenerator, generateRestrictedPartition_givenNAndK_returnListOfKNumbers) {
    for (int i=0; i<10; i++) {
        std::list<size_t> partition = sampleRandomRestrictedPartition(N, K);
        EXPECT_EQ(partition.size(), K);
    }
}


TEST(TestRestrictedPartitionGenerator, generateRestrictedPartition_givenNAndK_partitionSumsToN) {
    for (int i=0; i<10; i++) {
        std::list<size_t> partition = sampleRandomRestrictedPartition(N, K);
        size_t sum=0;
        for (size_t element: partition)
            sum += element;
        EXPECT_EQ(sum, N);
    }
}


TEST(TestRandomPermutationGenerator, generatePermutation_givenNk_permutationIsConsistentWithNk) {
    for (int i=0; i<10; i++) {
        std::list<size_t> nkList = sampleRandomComposition(N, K);
        std::vector<size_t> nk;
        for (auto _nk : nkList) nk.push_back(_nk);

        std::vector<size_t> permutation = sampleRandomPermutation(nk);

        EXPECT_EQ(permutation.size(), N);

        std::vector<size_t> actualNk(K, 0);
        for (auto i : permutation){
            ++actualNk[i];
        }
        for (size_t i = 0 ; i < K ; ++i){
            EXPECT_EQ(actualNk[i], nk[i]);
        }
    }
}

TEST(TestRandomMultinomialGenerator, generateMultinomial){
    seedWithTime();
    std::vector<double> p = {0.3, 0.4, 0.2, 0.1};
    size_t n = 1000;
    auto s = sampleMultinomial(n, p);

    double sum = 0;
    for (auto ss : s)
        sum += ss;
    EXPECT_EQ(sum, n);
}

TEST(TestRandomMultinomialGenerator, generateUniformMultinomial){
    seedWithTime();
    size_t n = 1000, k = 5;

    auto s = sampleUniformMultinomial(n, k);

    double sum = 0;
    for (auto ss : s)
        sum += ss;
    EXPECT_EQ(sum, n);
    EXPECT_EQ(s.size(), k);
}

}
