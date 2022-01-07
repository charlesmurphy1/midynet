#include "gtest/gtest.h"
#include <vector>

#include "FastMIDyNet/utility/maps.hpp"

namespace FastMIDyNet{

TEST(MapBaseClass, constructor_forIntKeysIntValues_returnMap){
    std::vector<int> keys = {0,1,2,3,4,5};
    std::vector<int> values = {0,1,2,3,4,5};
    Map<int, int> map(keys, values, 0);
    EXPECT_EQ(keys.size(), map.size());
}
TEST(MapBaseClass, copyconstructor_forIntKeysIntValues_returnMap){
    std::vector<int> keys = {0,1,2,3,4,5};
    std::vector<int> values = {0,1,2,3,4,5};
    Map<int, int> map(keys, values, 0);
    Map<int, int> otherMap(map);
    EXPECT_EQ(otherMap.size(), map.size());
}

TEST(MapBaseClass, constructor_forDefaultValue0_returnEmptyMap){
    Map<int, int> map(0);
    EXPECT_EQ(0, map.size());
}

TEST(MapBaseClass, get_forSomeEmptyKey_returnDefaultValue){
    Map<int, int> map(0);
    EXPECT_EQ(0, map.size());
    EXPECT_EQ(0, map.get(3));
    EXPECT_EQ(0, map.size());
}

TEST(MapBaseClass, set_forSomeEmptyKeyWithSomeValue){
    Map<int, int> map;
    EXPECT_EQ(0, map.size());
    map.set(3, 4);
    EXPECT_EQ(4, map.get(3));
    EXPECT_EQ(1, map.size());
}

TEST(MapBaseClass, operatorBraket_forEmptyKey_returnValue){
    Map<int, int> map;
    EXPECT_EQ(map[3], 0);
}

TEST(MapBaseClass, operatorBraket_forNonEmptyKey_returnValue){
    Map<int, int> map;
    map.set(3, 4);
    EXPECT_EQ(map[3], 4);
    EXPECT_EQ(map.get(3), 4);
}

TEST(MapBaseClass, isEmpty_forEmptyKey_returnTrue){
    Map<int, int> map;
    EXPECT_TRUE(map.isEmpty(3));
}

TEST(MapBaseClass, isEmpty_forNonEmptyKey_returnFalse){
    Map<int, int> map;
    map.set(3, 4);
    EXPECT_FALSE(map.isEmpty(3));
}

TEST(MapBaseClass, erase_forEmptyKey_doNothing){
    Map<int, int> map;
    map.set(3, 4);
    map.erase(10);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map[3], 4);
}

TEST(MapBaseClass, erase_forNonEmptyKey_eraseKey){
    Map<int, int> map;
    map.set(3, 4);
    map.erase(3);
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map[3], 0);
}

TEST(MapBaseClass, testing_maps_in_for_loops){
    Map<int, int> map;
    map.set(0, 4);
    map.set(1, 5);
    map.set(2, 36);
    map.set(3, 7);
}

TEST(MapBaseClass,operatorEqualEqual_forSameMap_returnTrue){
    Map<int, int> map1, map2;

    map1.set(0, 4); map2.set(0, 4);
    map1.set(1, 5); map2.set(1, 5);
    map1.set(2, 36); map2.set(2, 36);
    map1.set(3, 7); map2.set(3, 7);
    EXPECT_TRUE(map1 == map2);
}

TEST(MapBaseClass,operatorEqualEqual_forDifferentMap_returnFalse){
    Map<int, int> map1, map2;

    map1.set(0, 4); map2.set(0, 3); //difference here
    map1.set(1, 5); map2.set(1, 5);
    map1.set(2, 36); map2.set(2, 36);
    map1.set(3, 7); map2.set(3, 7);
    EXPECT_FALSE(map1 == map2);
}

TEST(IntMapClass, increment_forEmptyKey_keyIsNow1){
    IntMap<int> map;
    map.increment(3);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map[3], 1);
}

TEST(IntMapClass, increment_forNonEmptyKey_keyIsIncremented){
    IntMap<int> map;
    map.set(3, 4);
    map.increment(3);
    EXPECT_EQ(map[3], 5);
}

TEST(IntMapClass, increment_forNonEmptyKeyWithIncEqualToTwo_keyIsIncrementedTwice){
    IntMap<int> map;
    map.set(3, 4);
    map.increment(3, 2);
    EXPECT_EQ(map[3], 6);
}

TEST(CounterMapClass, increment_forEmptyKey_keyIsNow1){
    CounterMap<int> map;
    map.increment(3);
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map[3], 1);
}

TEST(CounterMapClass, increment_forNonEmptyKey_keyIsIncremented){
    CounterMap<int> map;
    map.set(3, 4);
    map.increment(3);
    EXPECT_EQ(map[3], 5);
}

TEST(CounterMapClass, decrement_forEmptyKey_doNothing){
    CounterMap<int> map;
    map.decrement(3);
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map[3], 0);
}

TEST(CounterMapClass, decrement_forNonEmptyKey_keyIsDecremented){
    CounterMap<int> map;
    map.set(3, 7);
    map.decrement(3);
    EXPECT_EQ(map[3], 6);
}

TEST(CounterMapClass, decrement_forNonEmptyKeyWithValue1_keyIsErased){
    CounterMap<int> map;
    map.set(3, 1);
    map.decrement(3);
    EXPECT_TRUE(map.isEmpty(3));
}

TEST(VectorOfMaps, erase_forSomeKey_noMemoryLoss){
    std::vector<CounterMap<size_t>> maps(10, 0);
    maps[0].increment(0);
    maps[1].increment(0);

    EXPECT_EQ(maps.size(), 10);
    maps.erase(maps.begin() + 1);
    EXPECT_EQ(maps.size(), 9);
    maps.erase(maps.begin());
    EXPECT_EQ(maps.size(), 8);
}

}
