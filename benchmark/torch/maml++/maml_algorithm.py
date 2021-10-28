#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Tuple
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import parl
from maml_model import MAMLModel
from config import Config


class MAML(parl.Algorithm):
    def __init__(self, model: MAMLModel, config: Config, device: torch.device):

        self.model = model
        self.learning_rates = nn.Parameter(
            data=torch.ones((config.num_updates_per_iter, config.num_layers),
                            device=device) * config.task_learning_rate,
            requires_grad=config.learnable_learning_rates)

        self.optimizer = optim.Adam(
            self.model.get_weights() + [self.learning_rates],
            lr=config.meta_learning_rate,
            amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=config.total_epochs,
            eta_min=config.min_learning_rate)
        self.config = config
        self.device = device

    def train_one_iter(self, data_batch: Tuple, current_epoch: int) -> float:
        '''Train model with a batch of tasks

        Args:
            data_batch: a batch of tasks, each task contains support set and query set
            current_epoch

        Returns:
            Average MSE loss on query set
        '''
        x_support_set, y_support_set, x_query_set, y_query_set = data_batch
        self.model.zero_grad()
        if self.config.learning_rate_scheduler:
            self.scheduler.step(current_epoch)
        batch_losses = []
        for x_support, y_support, x_query, y_query in zip(
                x_support_set, y_support_set, x_query_set, y_query_set):
            task_loss = 0
            per_step_loss_importance_vectors = self._get_per_step_loss_importance_vector(
                current_epoch)
            model_weights = self.model.get_weights()

            for num_step in range(self.config.num_updates_per_iter):

                support_loss = self._caculate_loss(x_support, y_support,
                                                   model_weights)

                model_weights = self._inner_loop_update(
                    loss=support_loss,
                    weights=model_weights,
                    use_second_order=self.config.second_order,
                    current_update_step=num_step)

                if self.config.use_multi_step_loss_optimization and current_epoch < self.config.multi_step_loss_num_epochs:
                    query_loss = self._caculate_loss(
                        x=x_query, y=y_query, weights=model_weights)
                    task_loss = task_loss + per_step_loss_importance_vectors[
                        num_step] * query_loss

                elif num_step == (self.config.num_updates_per_iter - 1):
                    query_loss = self._caculate_loss(
                        x=x_query, y=y_query, weights=model_weights)
                    task_loss = query_loss

            batch_losses.append(task_loss)

        avg_batch_loss = sum(batch_losses) / len(batch_losses)

        self._outer_loop_update(avg_batch_loss)

        return avg_batch_loss.item()

    def evaluate_one_iter(self, data_batch: Tuple) -> float:
        '''Evaluate current model with a batch of tasks

        Args:
            data_batch: a batch of tasks, each task contains support set and query set

        Returns:
            average MSE loss on query set
        '''
        x_support_set, y_support_set, x_query_set, y_query_set = data_batch
        self.model.zero_grad()
        test_losses = []
        for x_support, y_support, x_query, y_query in zip(
                x_support_set, y_support_set, x_query_set, y_query_set):

            model_weights = self.model.get_weights()

            for num_step in range(self.config.num_updates_per_iter):

                support_loss = self._caculate_loss(x_support, y_support,
                                                   model_weights)

                model_weights = self._inner_loop_update(
                    loss=support_loss,
                    weights=model_weights,
                    use_second_order=self.config.second_order,
                    current_update_step=num_step)

            with torch.no_grad():
                query_loss = self._caculate_loss(x_query, y_query,
                                                 model_weights)

            test_losses.append(query_loss.item())

        test_loss = sum(test_losses) / len(test_losses)

        return test_loss

    def _inner_loop_update(self, loss: torch.Tensor, weights: List,
                           use_second_order: bool,
                           current_update_step: int) -> List:
        ''' Applies an inner loop update

        Args:
            loss: Loss on support set
            weights: Network weights
            use_second_order: If use second order
            current_update_step

        Returns:
            updated network weights
        
        '''
        self.model.zero_grad(weights)

        grads = torch.autograd.grad(
            loss, weights, create_graph=use_second_order, allow_unused=True)

        updated_weights = list()

        for idx, (weight, grad) in enumerate(zip(weights, grads)):
            updated_weight = weight - self.learning_rates[current_update_step,
                                                          idx // 2] * grad
            updated_weights.append(updated_weight)

        return updated_weights

    def _outer_loop_update(self, loss):
        """Applies an outer loop update on the meta-parameters of the model."""
        self.optimizer.zero_grad()
        loss.backward()
        if self.learning_rates.requires_grad:
            self.learning_rates.grad.clamp_(-20, 20)
        self.optimizer.step()

    def _get_per_step_loss_importance_vector(self, epoch: int) -> torch.Tensor:
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.

        Returns:
            A tensor to be used to compute the weighted average of the loss, useful for
            the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(self.config.num_updates_per_iter
                               ) / self.config.num_updates_per_iter
        decay_rate = 1.0 / self.config.num_updates_per_iter / self.config.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.config.num_updates_per_iter

        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (epoch * decay_rate),
                                    min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        loss_weights[-1] = 1 - np.sum(loss_weights[:-1])

        loss_weights = torch.Tensor(loss_weights, device=self.device)
        return loss_weights

    def _caculate_loss(self, x: torch.Tensor, y: torch.Tensor,
                       weights: List) -> torch.Tensor:
        pred_y = self.model.forward(x, weights)

        loss = F.mse_loss(pred_y, y)

        return loss
