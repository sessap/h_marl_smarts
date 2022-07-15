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
from baselines.marl_benchmark.wrappers.rllib.frame_stack import FrameStack
from baselines.marl_benchmark.wrappers.rllib.frame_stack_halluc import H_FrameStack
from baselines.marl_benchmark.wrappers.rllib.frame_stack_eval import FrameStack_augm


class EarlyDone(FrameStack):
    def step(self, agent_actions):
        observations, rewards, dones, infos = super(EarlyDone, self).step(agent_actions)
        dones["__all__"] = any(list(dones.values()))
        return observations, rewards, dones, infos

class H_EarlyDone(H_FrameStack):
    def step(self, agent_actions):
        observations, rewards, dones, infos = super(H_EarlyDone, self).step(agent_actions)
        dones["__all__"] = any(list(dones.values()))
        return observations, rewards, dones, infos

class EarlyDone_augm(FrameStack_augm):
    def step(self, agent_actions, sv_next_states=None, get_rewards_fun=None):
        observations, rewards, dones, infos = super(EarlyDone_augm, self).step(agent_actions, sv_next_states=sv_next_states, get_rewards_fun=get_rewards_fun)
        dones["__all__"] = any(list(dones.values()))
        return observations, rewards, dones, infos