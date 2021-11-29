#ifndef FAST_MIDYNET_MAPS_H
#define FAST_MIDYNET_MAPS_H

#include <iostream>
#include <map>
#include <vector>

namespace FastMIDyNet{

template <typename KeyType, typename ValueType>
class Map{
public:

    Map(const std::vector<KeyType>& keys, const std::vector<ValueType>& values, ValueType defaultValue):
        m_map(), m_defaultValue(defaultValue){
        for (size_t k = 0; k < keys.size(); ++k)
            m_map.insert(std::pair<KeyType, ValueType>(keys[k], values[k]));
    }
    Map(const Map<KeyType, ValueType>& other):
        m_map(other.m_map), m_defaultValue(other.m_defaultValue){}
    Map(ValueType defaultValue):
        m_map(), m_defaultValue(defaultValue) { }
        Map():
            m_map(), m_defaultValue() { }
    size_t size(){
        return m_map.size();
    }

    const ValueType& operator[](KeyType key){ return get(key); }
    const ValueType& get(KeyType key) {
        if ( isEmpty(key) ) set(key, m_defaultValue);
        return m_map[key];
    }

    const ValueType& get(KeyType key) const {
        return m_map[key];
    }

    void set(KeyType key, ValueType value) {
        m_map[key] = value;
    }

    bool isEmpty(KeyType key) const{
        return m_map.count(key) == 0;
    }

    void erase(KeyType key){ if (not isEmpty(key)) { m_map.erase(key); }}
    typename std::map<KeyType, ValueType>::iterator begin() { return m_map.begin(); }
    typename std::map<KeyType, ValueType>::iterator end() { return m_map.end(); }

protected:
    std::map<KeyType, ValueType> m_map;
    ValueType m_defaultValue;
};

template < typename KeyType>
class IntMap: public Map<KeyType, int>{
public:
    IntMap(int defaultValue=0): Map<KeyType, int>(defaultValue) {}
    void increment(KeyType key, int inc=1){ this->set(key, this->get(key) + inc); }
    void decrement(KeyType key, int dec=1){ increment(key, -dec); }
};

template < typename KeyType>
class CounterMap: public Map<KeyType, size_t>{
public:
    CounterMap(size_t defaultValue=0): Map<KeyType, size_t>(defaultValue) {}
    void increment(KeyType key, int inc=1){
        if (static_cast<int>(this->get(key)) + inc <= 0) this->erase(key);
        else this->set(key, this->get(key) + inc);
    }

    void decrement(KeyType key, int dec=1){ increment(key, -dec); }

};


}
#endif
