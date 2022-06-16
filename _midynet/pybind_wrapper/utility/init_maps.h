#ifndef FAST_MIDYNET_PYWRAPPER_INIT_UTILITY_MAPS_H
#define FAST_MIDYNET_PYWRAPPER_INIT_UTILITY_MAPS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FastMIDyNet/utility/maps.hpp"

namespace py = pybind11;
namespace FastMIDyNet{

template <typename KeyType, typename ValueType>
py::class_< Map<KeyType, ValueType> > declareMap(py::module& m, std::string pyName){
    return py::class_<Map<KeyType, ValueType>>(m, pyName.c_str())
        .def(py::init<const std::vector<KeyType>&, const std::vector<ValueType>&, ValueType>(), py::arg("keys"), py::arg("values"), py::arg("default"))
        .def(py::init<const Map<KeyType, ValueType>&>(), py::arg("map"))
        .def(py::init<ValueType>(), py::arg("default"))
        .def("__getitem__", [](const Map<KeyType, ValueType>& self, KeyType key) { return self.get(key); })
        .def("__eq__", [](Map<KeyType, ValueType>& self, Map<KeyType, ValueType>& other) { return self.operator==(other); })
        .def("__iter__", [](Map<KeyType, ValueType> &self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
        .def("size", &Map<KeyType, ValueType>::size)
        .def("get", &Map<KeyType, ValueType>::get, py::arg("key"))
        .def("set", &Map<KeyType, ValueType>::set, py::arg("key"), py::arg("value"))
        .def("is_empty", &Map<KeyType, ValueType>::isEmpty, py::arg("key"))
        .def("erase", &Map<KeyType, ValueType>::erase, py::arg("key"))
        .def("clear", &Map<KeyType, ValueType>::clear)
        .def("display", &Map<KeyType, ValueType>::display)
        .def("get_keys", &Map<KeyType, ValueType>::getKeys)
        .def("get_values", &Map<KeyType, ValueType>::getValues)
        ;
}

template <typename KeyType>
py::class_<IntMap<KeyType>, Map<KeyType, int>> declareIntMap(py::module& m, std::string pyName){
    return py::class_<IntMap<KeyType>, Map<KeyType, int>>(m, pyName.c_str())
        .def(py::init<const std::vector<KeyType>&, const std::vector<int>&, int>(), py::arg("keys"), py::arg("values"), py::arg("default")=0)
        .def(py::init<const IntMap<KeyType>&>(), py::arg("map"))
        .def(py::init<int>(), py::arg("default")=0)
        .def("increment", &IntMap<KeyType>::increment, py::arg("key"), py::arg("inc")=1)
        .def("decrement", &IntMap<KeyType>::decrement, py::arg("key"), py::arg("dec")=1)
        ;
}

template <typename KeyType>
py::class_<CounterMap<KeyType>, Map<KeyType, size_t>> declareCounterMap(py::module& m, std::string pyName){
    return py::class_<CounterMap<KeyType>, Map<KeyType, size_t>>(m, pyName.c_str())
        .def(py::init<const std::vector<KeyType>&, const std::vector<size_t>&, size_t>(), py::arg("keys"), py::arg("values"), py::arg("default")=0)
        .def(py::init<const CounterMap<KeyType>&>(), py::arg("map"))
        .def(py::init<size_t>(), py::arg("default")=0)
        .def("increment", &CounterMap<KeyType>::increment, py::arg("key"), py::arg("inc")=1)
        .def("decrement", &CounterMap<KeyType>::decrement, py::arg("key"), py::arg("dec")=1)
        .def("get_sum", &CounterMap<KeyType>::getSum)
        ;
}


void initMaps(py::module& m){
    declareMap<size_t, int>(m, "UnIntKeyedIntValuedMap");
    declareIntMap<size_t>(m, "UnIntKeyedIntMap");

    declareMap<size_t, size_t>(m, "UnIntKeyedUnIntValuedMap");
    declareCounterMap<size_t>(m, "UnIntKeyedCounterMap");
}

}

#endif
