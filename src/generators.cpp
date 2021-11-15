#include <random>
#include <vector>

#include "FastMIDyNet/generators.h"
#include "FastMIDyNet/types.h"


using namespace std;
namespace FastMIDyNet {
int generateCategorical(const vector<double>& probs, RNG& rng){
    discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}
}
