#ifndef FAST_MIDYNET_MAPS_H
#define FAST_MIDYNET_MAPS_H

#include <iostream>
#include <map>
#include <list>
#include <string>
#include <sstream>
#include <vector>

namespace FastMIDyNet{

template <typename KeyType, typename ValueType>
class Map{
public:

    Map(const std::vector<KeyType>& keys, const std::vector<ValueType>& values, ValueType defaultValue):
        m_map(), m_defaultValue(defaultValue){
        for (size_t k = 0; k < keys.size(); ++k){
            m_map.insert(std::pair<KeyType, ValueType>(keys[k], values[k]));
        }
    }
    Map(const Map<KeyType, ValueType>& other):
        m_map(other.m_map), m_defaultValue(other.m_defaultValue){}
    Map(ValueType defaultValue):
        m_map(), m_defaultValue(defaultValue) { }
    Map():
        m_map(), m_defaultValue() { }
    size_t size() const {
        return m_map.size();
    }

    const ValueType& operator[](const KeyType& key) const { return get(key); }
    bool operator==(const Map<KeyType, ValueType> rhs){
        for( auto k : *this){
            if (rhs.isEmpty(k.first) or rhs.get(k.first) != k.second) return false;
        }
        return true;
    }
    const std::list<KeyType>& keys() const{
        std::list<KeyType> k;
        for (auto x: m_map)
            k.push_back(x.first);
        return k;
    }
    const std::list<ValueType> values() const{
        std::list<ValueType> v;
        for (auto x: m_map)
            v.push_back(x.second);
        return v;
    }
    const ValueType& get(const KeyType& key) const {
        if ( isEmpty(key) ) return m_defaultValue;
        else return m_map.at(key);
    }

    void set(const KeyType& key, const ValueType& value) {
        m_map[key] = value;
    }

    bool isEmpty(KeyType key) const{
        return m_map.count(key) == 0;
    }

    void erase(KeyType key){ if (not isEmpty(key)) m_map.erase(key); }
    void clear(){ m_map.clear(); }
    typename std::map<KeyType, ValueType>::iterator begin() { return m_map.begin(); }
    typename std::map<KeyType, ValueType>::iterator end() { return m_map.end(); }

    std::string display() const{
        std::stringstream ss;
        for (auto k : m_map){
            ss << "(" << std::to_string(k.first) << " -> " << std::to_string(k.second) << ") ";
        }
        return ss.str();
    }

protected:
    std::map<KeyType, ValueType> m_map;
    ValueType m_defaultValue;
};

template < typename KeyType>
class IntMap: public Map<KeyType, int>{
public:
    IntMap(int defaultValue=0): Map<KeyType, int>(defaultValue) {}
    IntMap(const IntMap<KeyType>& other): Map<KeyType, int>(other) {}
    IntMap(const std::vector<KeyType>& keys, const std::vector<int>& values, int defaultValue=0):
        Map<KeyType, int>(keys, values, defaultValue){}
    void increment(KeyType key, int inc=1){ this->set(key, this->get(key) + inc); }
    void decrement(KeyType key, int dec=1){ increment(key, -dec); }
};

template < typename KeyType>
class CounterMap: public Map<KeyType, size_t>{
public:
    CounterMap(size_t defaultValue=0): Map<KeyType, size_t>(defaultValue) {}
    CounterMap(const CounterMap<KeyType>& other): Map<KeyType, size_t>(other) {}
    CounterMap(const std::vector<KeyType>& keys, const std::vector<size_t>& values, size_t defaultValue=0):
        Map<KeyType, size_t>(keys, values, defaultValue){}
    void increment(KeyType key, int inc=1){
        if (static_cast<int>(this->get(key)) + inc <= 0) this->erase(key);
        else this->set(key, this->get(key) + inc);
    }

    void decrement(KeyType key, int dec=1){ increment(key, -dec); }

};

template < typename T1, typename T2 >
class OrderedPair: public std::pair<T1, T2>{
public:
    OrderedPair(T1 first, T2 second):std::pair<T1, T2>(first, second){
        if (first > second){
            auto temp = this->first ;
            this->first = this->second ;
            this->second = temp;
        }
    }
};


}
#endif
