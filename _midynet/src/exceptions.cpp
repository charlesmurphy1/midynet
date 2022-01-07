#include <stdexcept>
#include "FastMIDyNet/exceptions.h"

void assertValidProbability(double probability) {
    if (probability > 1 || probability < 0)
        throw std::invalid_argument("Invalid probability "+std::to_string(probability)+
                ". Probability must be contained between 0 and 1.");
}
