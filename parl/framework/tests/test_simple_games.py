#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.computation_task import ComputationTask
from parl.algorithm_zoo.simple_algorithms import SimpleAC, SimpleQ
from parl.model_zoo.simple_models import SimpleModelAC, SimpleModelQ
import numpy as np
import unittest
import math
import gym


def unpack_exps(exps):
    return [np.array(l).astype('int' if i==2 else 'float32') \
            for i, l in enumerate(zip(*exps))]


def sample(past_exps, n):
    indices = np.random.choice(len(past_exps), n)
    return [past_exps[i] for i in indices]


class TestGymGame(unittest.TestCase):
    def test_gym_games(self):
        """
        Test games in OpenAI gym.
        """

        games = ["MountainCar-v0", "CartPole-v0"]
        final_rewards_thresholds = [
            -1.8,  ## drive to the right top in 180 steps (timeout is -2.0)
            1.5  ## hold the pole for at least 150 steps
        ]

        mlp_layer_confs = [
            dict(
                size=128, act="relu"),
            dict(
                size=128, act="relu"),
            dict(
                size=128, act="relu"),
        ]

        for game, threshold in zip(games, final_rewards_thresholds):
            for on_policy in [False, True]:

                if on_policy and game != "CartPole-v0":
                    ## SimpleAC has difficulty training mountain-car and acrobot
                    continue

                env = gym.make(game)
                state_shape = env.observation_space.shape[0]
                num_actions = env.action_space.n

                if on_policy:
                    alg = SimpleAC(
                        model=SimpleModelAC(
                            dims=state_shape,
                            num_actions=num_actions,
                            mlp_layer_confs=mlp_layer_confs +
                            [dict(
                                size=num_actions, act="softmax")]),
                        hyperparas=dict(lr=1e-3))
                else:
                    alg = SimpleQ(
                        model=SimpleModelQ(
                            dims=state_shape,
                            num_actions=num_actions,
                            mlp_layer_confs=mlp_layer_confs +
                            [dict(size=num_actions)]),
                        hyperparas=dict(lr=1e-4),
                        exploration_end_batches=25000,
                        update_ref_interval=100)

                print "algorithm: " + alg.__class__.__name__

                ct = ComputationTask(algorithm=alg)
                batch_size = 16
                if not on_policy:
                    train_every_steps = batch_size / 4
                    buffer_size_limit = 100000

                max_episode = 5000

                average_episode_reward = []
                past_exps = []
                max_steps = env._max_episode_steps
                for n in range(max_episode):
                    ob = env.reset()
                    episode_reward = 0
                    for t in range(max_steps):
                        res, _ = ct.predict(inputs=dict(sensor=np.array(
                            [ob]).astype("float32")))
                        pred_action = res["action"][0][0]

                        next_ob, reward, next_is_over, _ = env.step(
                            pred_action)
                        reward /= 100
                        episode_reward += reward

                        past_exps.append((ob, next_ob, [pred_action],
                                          [reward], [not next_is_over]))
                        ## only for off-policy training we use a circular buffer
                        if (not on_policy
                            ) and len(past_exps) > buffer_size_limit:
                            past_exps.pop(0)

                        ## compute the learning condition
                        learn_cond = False
                        if on_policy:
                            learn_cond = (len(past_exps) >= batch_size)
                            exps = past_exps  ## directly use all exps in the buffer
                        else:
                            learn_cond = (
                                t % train_every_steps == train_every_steps - 1)
                            exps = sample(past_exps,
                                          batch_size)  ## sample some exps

                        if learn_cond:
                            sensor, next_sensor, action, reward, next_episode_end \
                                = unpack_exps(exps)
                            cost = ct.learn(
                                inputs=dict(sensor=sensor),
                                next_inputs=dict(next_sensor=next_sensor),
                                next_episode_end=dict(
                                    next_episode_end=next_episode_end),
                                actions=dict(action=action),
                                rewards=dict(reward=reward))
                            ## we clear the exp buffer for on-policy
                            if on_policy:
                                past_exps = []

                        ob = next_ob

                        ## end before the Gym wrongly gives game_over=True for a timeout case
                        if t == max_steps - 2 or next_is_over:
                            break

                    if n % 50 == 0:
                        print("episode reward: %f" % episode_reward)

                    average_episode_reward.append(episode_reward)
                    if len(average_episode_reward) > 20:
                        average_episode_reward.pop(0)

                ### compuare the average episode reward to reduce variance
                self.assertGreater(
                    sum(average_episode_reward) / len(average_episode_reward),
                    threshold)


if __name__ == "__main__":
    unittest.main()
