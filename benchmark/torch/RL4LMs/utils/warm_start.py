import os
from typing import Any, Dict

import torch

# from rl4lms.envs.text_generation.logging_utils import Tracker
# from rl4lms.envs.text_generation.policy.base_policy import LMActorCriticPolicy


class ActorCriticWarmStartMixin:
    def get_state_dict(self) -> Dict[str, Any]:
        state_dict = {
            "policy_model": self._policy_model.state_dict(),
            "value_model": self._value_model.state_dict(),
            "value_head": self._value_head.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        return state_dict

    def load_from_dict(self, state_dict: dict = None):
        if state_dict is not None:
            self._policy_model.load_state_dict(state_dict["policy_model"])
            self._value_model.load_state_dict(state_dict["value_model"])
            self._value_head.load_state_dict(state_dict["value_head"])
            self.optimizer.load_state_dict(state_dict["optimizer"])



class OnPolicyWarmStartMixin:
    def get_state_dict(self) -> Dict[str, Any]:
        # just the kl controller state is sufficient for onpolicy algs
        state_dict = {
            "kl_controller_state": self._kl_controller.get_state_dict(),
        }
        return state_dict

    def load_from_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if state_dict is not None:
            self._kl_controller.load_from_state_dict(
                state_dict["kl_controller_state"])

# ################## Policy Warm Start Mixins#######################################
#
#
# class ActorOnlyWarmStartMixin:
#     def get_state_dict(self) -> Dict[str, Any]:
#         state_dict = {
#             "policy_model": self._policy_model.state_dict(),
#             "optimizer": self.optimizer.state_dict()
#         }
#         return state_dict
#
#     def load_from_dict(self, state_dict: dict = None):
#         if state_dict is not None:
#             self._policy_model.load_state_dict(state_dict["policy_model"])
#             self.optimizer.load_state_dict(state_dict["optimizer"])
#
#
#
#
#
#
#
# ################## Algorithm Warm Start Mixins#######################################

#
#
# class OffPolicyWarmStartMixin:
#     def get_state_dict(self) -> Dict[str, Any]:
#         # TBD: just buffer is sufficient? or is there something else?
#         state_dict = {
#             "replay_buffer": self.replay_buffer.get_state_dict(),
#         }
#         return state_dict
#
#     def load_from_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
#         if state_dict is not None:
#             self.replay_buffer.load_from_state_dict(
#                 state_dict["replay_buffer"])
#
#
# ################## Trainer Warm Start Mixins#######################################
# class TrainerWarmStartMixin:
#     def _get_recent_ckpt_path(self, tracker: Tracker):
#         try:
#             checkpoints = os.listdir(tracker.checkpoint_base_path)
#         except:
#             os.makedirs(tracker.checkpoint_base_path)
#             checkpoints = os.listdir(tracker.checkpoint_base_path)
#
#         if len(checkpoints) == 0:
#             return None, None
#
#         sorted_ckpts = sorted(checkpoints, reverse=True,
#                               key=lambda ckpt: int(ckpt.split("_")[1]))
#         recent_ckpt = sorted_ckpts[0]
#         recent_ckpt_id = int(recent_ckpt.split("_")[1])
#
#         recent_ckpt_path = os.path.join(
#             tracker.checkpoint_base_path, f"checkpoint_{recent_ckpt_id}")
#         return recent_ckpt_path, recent_ckpt_id
#
#     def load_trainer_state(self, tracker: Tracker):
#         recent_ckpt_path, _ = self._get_recent_ckpt_path(tracker)
#         state_dict = None
#         try:
#             if recent_ckpt_path is not None:
#                 state_dict = torch.load(
#                     recent_ckpt_path, map_location=torch.device("cuda"))
#                 tracker.log_info("Model checkpoint found - Warm starting")
#                 self._policy_state_dict = state_dict["policy_state"]
#                 self._alg_state_dict = state_dict["alg_state"]
#                 self._trainer_state = state_dict["trainer_state"]
#
#                 tracker.log_info(
#                     f"Loaded the current trainer state from: {self._trainer_state}")
#             else:
#                 self._policy_state_dict = None
#                 self._alg_state_dict = None
#                 self._trainer_state = {
#                     "current_iter": 0,
#                 }
#         except Exception as e:
#             tracker.log_info(f"Exception while doing warm start {e}")
#             tracker.log_info(
#                 f"Checkpoint may be corrupted...skipping warm start")
#             self._policy_state_dict = None
#             self._alg_state_dict = None
#             self._trainer_state = {
#                 "current_iter": 0,
#             }
#
#     def save_trainer_state(self, tracker: Tracker,
#                            policy: LMActorCriticPolicy,
#                            trainer_state: Dict[str, Any]):
#         full_state = {
#             "alg_state": self._alg.get_state_dict(),
#             "policy_state": policy.get_state_dict(),
#             "trainer_state": trainer_state
#         }
#         _, recent_ckpt_id = self._get_recent_ckpt_path(tracker)
#
#         # hot fix - just to save only the last checkpoint (overwrite)
#         new_ckpt_id = 0 if recent_ckpt_id is None else recent_ckpt_id + 1
#         new_ckpt_path = os.path.join(
#             tracker.checkpoint_base_path, f"checkpoint_{new_ckpt_id}")
#         torch.save(full_state, new_ckpt_path, pickle_protocol=4)
