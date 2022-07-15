# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import numpy as np
from ray.rllib.utils.annotations import override

from baselines.marl_benchmark.wrappers.rllib.frame_stack import FrameStack
from baselines.marl_benchmark.sv_models.sv_models import SV_GPModel
import copy

class H_FrameStack(FrameStack):
    """ By default, this wrapper will stack 3 consecutive frames as an agent observation"""

    def __init__(self, config):
        super(H_FrameStack, self).__init__(config)

        # initialize sv model
        data_path = config["custom_config"]["model_path"] + '/training_data.pth'
        model_state_path = config["custom_config"]["model_path"] + '/model_state.pth'
        train_x, train_y = torch.load(data_path)
        self.sv_model = SV_GPModel(train_x, train_y)
        self.sv_model.load_model(model_state_path)

        self.sv_states = {}
        self.num_samples = config["custom_config"]["num_samples"]
        self.unif_grid = config["custom_config"]["unif_grid"]
        self.thompson_sampling = config["custom_config"]["thompson_sampling"]
        self.beta_t = config["custom_config"]["beta_t"]

    @override(FrameStack)
    def _get_rewards(self, env_observations, env_rewards, addframe=False):
        agent_ids = list(env_rewards.keys())
        rewards = dict.fromkeys(agent_ids, None)

        obs_frames = copy.deepcopy(self.frames)
        for k, frame in env_observations.items():
            # if last frame is not env_observations, then add it to self.frames:
            if addframe:  # obs_frames[k][-1] is not frame:
                obs_frames[k].append(frame)
            rewards[k] = self.reward_adapter(list(obs_frames[k]), env_rewards[k])
        return rewards

    @override(FrameStack)
    def step(self, agent_actions):
        agent_actions = {
            agent_id: self.action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        # compute realization of next sv states based on current model.
        # These are fed into the hallucinated env.step function
        if self.sv_model and self.sv_states:
            sv_next_states_realiz = []
            if not self.thompson_sampling:
                etas = [np.zeros(2)]
                for i in range(self.num_samples-1):
                    if self.unif_grid:
                        eta = -1 + 2*i/(self.num_samples-1)
                        etas.append(np.repeat(eta,2)*self.beta_t)
                    else:
                        etas.append(np.random.uniform(low=-1, high=1, size=(2))*self.beta_t)
                for e in etas:
                    sv_next_states = {}
                    for sv_id in self.sv_states:
                        sv_next_states[sv_id] = self.sv_model.predict_state(torch.from_numpy(np.array([self.sv_states[sv_id]])),
                                                                   etas=e)
                    sv_next_states_realiz.append(sv_next_states)
            else:
                # Thompson sampling exploration
                sv_next_states = {}
                for sv_id in self.sv_states:
                    sv_next_states[sv_id] = self.sv_model.predict_state(torch.from_numpy(np.array([self.sv_states[sv_id]])), thompson_sample=True)
                sv_next_states_realiz.append(sv_next_states)
        else:
            sv_next_states_realiz = None

        env_observations, env_rewards, dones, infos = super(FrameStack, self).step(
            agent_actions, sv_next_states=sv_next_states_realiz, get_rewards_fun=self._get_rewards
        )

        # update states of all sv_id in env_observation
        if self.sv_model:
            self.sv_states = {}
            sv_ids = list(self.sv_model.compute_sv_ego_states(env_observations).keys())
            for sv_id in sv_ids:
                self.sv_states[sv_id] = self.sv_model.compute_sv_state(sv_id, env_observations)


        observations = self._get_observations(env_observations)
        rewards = self._get_rewards(env_observations, env_rewards)
        infos = self._get_infos(env_observations, env_rewards, infos)
        self._update_last_observation(self.frames)

        return observations, rewards, dones, infos