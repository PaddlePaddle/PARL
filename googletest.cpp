/*************************************************************************
    > File Name: googletest.cpp
    > Author:likejiao 
    > Mail: likejiao@baidu.com
    > Created Time: 五  4/19 10:33:24 2024
    > Usage: 
 ************************************************************************/

#include<iostream>
#include <gtest/gtest.h>
 
TEST(SimpleTest, TestEqual) {
  EXPECT_EQ(1, 1);
}
 
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

