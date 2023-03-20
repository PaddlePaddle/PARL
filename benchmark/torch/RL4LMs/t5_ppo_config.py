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

config = {
    'tokenizer': {
        'model_name': 't5-base',
        'padding_side': 'left',
        'truncation_side': 'left',
        'pad_token_as_eos_token': False
    },
    'datapool': {
        'id': 'cnn_daily_mail',
        'prompt_prefix': 'Summarize: '
    },
    'instructor': {
        'parl_master_address': 'localhost:8811',
        'n_instructors': 10,
        'reward_fn': {
            'rouge_type': 'rouge1'
        },
        'max_prompt_length': 512,
        'max_episode_length': 100,
        'terminate_on_eos': True,
        'prompt_truncation_side': 'right',
        'context_start_token': 0
    },
    'kl_div': {
        'coeff': 0.001,
        'target_kl': 0.2
    },
    'rollout_buffer': {
        'n_steps_per_instructor': 512  # buffer length = n_steps_per_instructor * n_instructors
    },
    'agent': {
        'batch_size': 32,
        'n_epochs': 5,
        'alg': {
            'initial_lr': 0.000002,
            'entropy_coef': 0.0
        },
        'model': {
            'model_name': 't5-base',
            'apply_model_parallel': True,
            'prompt_truncation_side': 'right',
            'generation_kwargs': {
                'do_sample': True,
                'top_k': 50,
                'min_length': 50,
                'max_new_tokens': 100
            }
        }
    },
    'examiner': {
        'max_prompt_length': 512,
        'eval_batch_size': 100,
        'generation_kwargs': {
            'do_sample': True,
            'top_k': 0,
            'temperature': 0.7,
            'min_length': 50,
            'max_new_tokens': 100
        },
        # metric list, each (id, args) is one metric
        'metrics': [{
            'id': 'meteor',
            'args': {}
        }, {
            'id': 'rouge'
        }, {
            'id': 'bleu',
            'args': {}
        }, {
            'id': 'bert_score',
            'args': {
                'language': 'en'
            }
        }, {
            'id': 'diversity',
            'args': {}
        }]
    },
    'train_evaluation': {
        'n_iters': 100,
        'eval_every': 10
    }
}
