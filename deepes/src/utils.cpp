#include "utils.h"

namespace DeepES {

void compute_centered_ranks(std::vector<float> &reward) {
  std::vector<std::pair<float, int>> reward_index;
  float gap = 1.0 / (reward.size() - 1);
  float normlized_rank = -0.5;
  int id = 0;
  for (auto& rew: reward) {
    reward_index.push_back(std::make_pair(rew, id));
    ++id;
  }
  std::sort(reward_index.begin(), reward_index.end());
  for (int i = 0; i < reward.size(); ++i) {
    id = reward_index[i].second;
    reward[id] = normlized_rank;
    normlized_rank += gap;
  }
}

}//namespace
