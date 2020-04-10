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
#include "evo_kit/optimizer_factory.h"
#include <memory>

namespace evo_kit {

TEST(SGDOptimizersTest, Method_update) {
    std::shared_ptr<EvoKitConfig> config = std::make_shared<EvoKitConfig>();
  auto optimizer_config = config->mutable_optimizer();
  optimizer_config->set_base_lr(1.0);
  optimizer_config->set_type("sgd");
  std::shared_ptr<Optimizer> optimizer = create_optimizer(config->optimizer());
  float sgd_wei[10]  = { 0.0       , 0.0       , 0.04216444, 0.0511456 , 0.04231584, 0.01089015, 0.06569759, 0.00127421,-0.00092832, 0.01128081};
  float sgd_grad[10] = {-0.11992419,-0.0       , 0.07681337,-0.06616384, 0.00249889, 0.01158612,-0.3067452 , 0.36048946,-0.15820622,-0.20014143};
  float sgd_new[10]  = { 0.01199242, 0.0       , 0.0344831 , 0.05776198, 0.04206595, 0.00973154, 0.09637211,-0.03477474, 0.014892306, 0.03129495};

  EXPECT_TRUE(optimizer->update(sgd_wei, sgd_grad, 10, "fc1"));
  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(sgd_new[i], sgd_wei[i]) << " i: " << i ;
  }
  EXPECT_TRUE(optimizer->update(sgd_wei, sgd_grad, 10, "fc1"));
  EXPECT_FALSE(optimizer->update(sgd_wei, sgd_grad, 9, "fc1"));
}

TEST(AdamOptimizersTest, Method_update) {
    std::shared_ptr<EvoKitConfig> config = std::make_shared<EvoKitConfig>();
  auto optimizer_config = config->mutable_optimizer();
  optimizer_config->set_base_lr(1.0);
  optimizer_config->set_type("adam");
  std::shared_ptr<Optimizer> optimizer = create_optimizer(config->optimizer());
  float adam_wei[10]  = { 0.0       , 0.0       , 0.04216444, 0.0511456 , 0.04231584, 0.01089015, 0.06569759, 0.00127421,-0.00092832, 0.01128081};
  float adam_grad[10] = {-0.11992419,-0.0       , 0.07681337,-0.06616384, 0.00249889, 0.01158612,-0.3067452 , 0.36048946,-0.15820622,-0.20014143};
  float adam_new[10]  = { 0.99999736, 0.        ,-0.95783144, 1.05114082,-0.95755763,-0.98908256, 1.06569656,-0.99872491, 0.99906968, 1.01127923};

  EXPECT_TRUE(optimizer->update(adam_wei, adam_grad, 10, "fc1"));
  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(adam_new[i], adam_wei[i]) << " i: " << i ;
  }
  EXPECT_TRUE(optimizer->update(adam_wei, adam_grad, 10, "fc1"));
  EXPECT_FALSE(optimizer->update(adam_wei, adam_grad, 9, "fc1"));
}

} // namespace

