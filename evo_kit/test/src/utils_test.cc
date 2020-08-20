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
#include "evo_kit/utils.h"

namespace evo_kit {

// Tests that the Utils::compute_centered_rank() method.
TEST(UtilsTest, Method_compute_centered_ranks) {
  float a[5] = {9.0, 8.0, 7.0, 6.0, 5.0};
  std::vector<float> reward_vec(a, a+5);
  EXPECT_EQ(compute_centered_ranks(reward_vec), true);
}


} // namespace

