#include <chrono>

#include "FastMIDyNet/rng.h"
#include "FastMIDyNet/types.h"


namespace FastMIDyNet {

RNG rng;

void setSeed(size_t seed){ rng.seed(seed); }
void setSeedWithTime(){
    setSeed(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

}
