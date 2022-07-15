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
from ray.rllib.utils.annotations import override

from baselines.marl_benchmark.wrappers.rllib.frame_stack import FrameStack
import copy


class FrameStack_augm(FrameStack):
    """ By default, this wrapper will stack 3 consecutive frames as an agent observation"""

    def __init__(self, config):
        super(FrameStack_augm, self).__init__(config)

    @override(FrameStack)
    def _get_rewards(self, env_observations, env_rewards, addframe=False):
        agent_ids = list(env_rewards.keys())
        rewards = dict.fromkeys(agent_ids, None)

        obs_frames = copy.deepcopy(self.frames)
        for k, frame in env_observations.items():
            # if last frame is not env_observations, then add it to self.frames:
            if addframe:#obs_frames[k][-1] is not frame:
                obs_frames[k].append(frame)
            rewards[k] = self.reward_adapter(list(obs_frames[k]), env_rewards[k])
        return rewards

    @override(FrameStack)
    def step(self, agent_actions, sv_next_states=None, get_rewards_fun=None):
        agent_actions = {
            agent_id: self.action_adapter(action)
            for agent_id, action in agent_actions.items()
        }
        env_observations, env_rewards, dones, infos = super(FrameStack, self).step(
            agent_actions, sv_next_states=sv_next_states, get_rewards_fun=get_rewards_fun
        )

        observations = self._get_observations(env_observations)
        rewards = self._get_rewards(env_observations, env_rewards)
        infos = self._get_infos(env_observations, env_rewards, infos)
        self._update_last_observation(self.frames)

        return observations, rewards, dones, infos, env_observations

    @override(FrameStack)
    def reset(self):
        observations = super(FrameStack, self).reset()
        for k, observation in observations.items():
            _ = [self.frames[k].append(observation) for _ in range(self.num_stack)]
        self._update_last_observation(self.frames)
        return self._get_observations(observations), observations

