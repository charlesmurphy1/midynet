#ifndef FAST_MIDYNET_COLLECTOR_HPP
#define FAST_MIDYNET_COLLECTOR_HPP

#include <vector>

#include "callback.h"

namespace FastMIDyNet{

template<typename T>
class Collector: public CallBack{
protected:
    std::vector<T> m_collected;

};

}

#endif
