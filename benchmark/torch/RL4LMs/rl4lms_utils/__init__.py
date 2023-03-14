#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .data_wrapper import RefPolicyOutput, GenerationInputs, GenerationOutputs,\
    PolicyType, Sample, Observation, TransitionInfo

from .huggingface_generation_util import override_generation_routines

from .buffer import DictRolloutBuffer

from .kl_controller import KLController

from .examiner import Examiner

from .data_pool import CNNDailyMail

from .reward_util import RougeRewardFunction

from .component_build_util import build_tokenizer, build_metrics, build_reward_fn,\
    build_datapool

from .rollout_util import RolloutUtil
