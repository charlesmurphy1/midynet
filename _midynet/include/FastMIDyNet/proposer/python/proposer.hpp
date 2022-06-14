#ifndef FAST_MIDYNET_PYTHON_PROPOSER_HPP
#define FAST_MIDYNET_PYTHON_PROPOSER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FastMIDyNet/proposer/proposer.hpp"
#include "FastMIDyNet/python/rv.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

template<typename MoveType,typename BaseClass = Proposer<MoveType>>
class PyProposer: public PyNestedRandomVariable<BaseClass>{
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;

    /* Pure abstract methods */
    MoveType proposeMove() const override { PYBIND11_OVERRIDE_PURE(MoveType, BaseClass, proposeMove, ); }

    /* Abstract & overloaded methods */
    ~PyProposer() override = default;
    void clear() override { PYBIND11_OVERRIDE(void, BaseClass, clear, ); }

};

}

#endif
