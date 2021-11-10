#include <random>
#include <vector>

#include "FastMIDyNet/dynamics/generators.h"
#include "FastMIDyNet/dynamics/types.h"

using namespace std;
namespace FastMIDyNet {
    int generateCategorical(vector<double> probs, RNGType rng){
        discrete_distribution dist(probs.begin(), probs.end());
        return dist(rng);
    }
}
