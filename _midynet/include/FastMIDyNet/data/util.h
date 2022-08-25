#ifndef FAST_MIDYNET_DYNAMICS_UTIL_H
#define FAST_MIDYNET_DYNAMICS_UTIL_H

#include <cmath>

namespace FastMIDyNet{

static inline double sigmoid(double x) {
    return 1./(1.+exp(-x));
}

}

#endif
