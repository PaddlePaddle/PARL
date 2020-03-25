//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include <vector>
#include "utils.h"

namespace DeepES {

// The fixture for testing class or file Utils.
class UtilsTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if their bodies would
  // be empty.

  UtilsTest() {
     // You can do set-up work for each test here.
  }

  ~UtilsTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
     // Code here will be called immediately after the constructor (right
     // before each test).
  }

  void TearDown() override {
     // Code here will be called immediately after each test (right
     // before the destructor).
  }

  // Class members declared here can be used by all tests in the test suite
  // for Utils.
};

// Tests that the Utils::compute_centered_rank() method.
TEST_F(UtilsTest, Method_compute_centered_ranks) {
  float a[5] = {9.0, 8.0, 7.0, 6.0, 5.0};
  std::vector<float> reward_vec(a, a+5);
  EXPECT_EQ(compute_centered_ranks(reward_vec), true);
}

// // Tests that Utils another method Xyz.
// TEST_F(UtilsTest, Method_Xyz) {
//   // Exercises the Method Xyz of Utils.
// }


}  // namespace project

