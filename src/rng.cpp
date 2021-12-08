#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {

RNG rng;

void setSeed(size_t seed){ rng.seed(seed); }

}
