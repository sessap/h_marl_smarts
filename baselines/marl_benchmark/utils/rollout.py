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
import collections
import torch
import numpy as np

from ray import logger
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluate import DefaultMapping, default_policy_agent_mapping
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray


def rollout(trainer, env_name, metrics_handler, num_steps, num_episodes, log_dir, sv_model=None, saved_transitions_dir='None'):
    """Reference: https://github.com/ray-project/ray/blob/master/rllib/rollout.py"""
    policy_agent_mapping = default_policy_agent_mapping
    assert hasattr(trainer, "workers") and isinstance(trainer.workers, WorkerSet)
    env = trainer.workers.local_worker().env
    multiagent = isinstance(env, MultiAgentEnv)
    if trainer.workers.local_worker().multiagent:
        policy_agent_mapping = trainer.config["multiagent"]["policy_mapping_fn"]
    policy_map = trainer.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    saved_transitions = {}
    for episode in range(num_episodes):
        saved_transitions[episode] = {}
        saved_transitions[episode]['env_obs'] = []
        saved_transitions[episode]['actions'] = []

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs, env_obs = env.reset()
        saved_transitions[episode]['env_obs'].append(env_obs)
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        step = 0
        sv_states = {}
        while not done and step < num_steps:
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = trainer.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = trainer.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]

            saved_transitions[episode]['actions'].append(action)

            if sv_model and sv_states:
                num_samples = 5
                sv_next_states_realiz = []
                for i in range(num_samples):
                    sv_next_states = {}
                    for sv_id in sv_states:
                        etas = np.random.uniform(low = -1, high=1, size=(2))*0.5
                        sv_next_states[sv_id] = sv_model.predict_state(torch.from_numpy(np.array([sv_states[sv_id]])), etas = etas)
                    sv_next_states_realiz.append(sv_next_states)
            else:
                sv_next_states_realiz = None
            next_obs, reward, done, info, env_obs = env.step(action, sv_next_states=sv_next_states_realiz, get_rewards_fun=env._get_rewards)
            saved_transitions[episode]['env_obs'].append(env_obs)
            if sv_model:
                sv_states = {}
                sv_ids = list(sv_model.compute_sv_ego_states(env_obs).keys())
                for sv_id in sv_ids:
                    sv_states[sv_id] = sv_model.compute_sv_state(sv_id, env_obs)

            metrics_handler.log_step(
                episode=episode,
                observations=multi_obs,
                actions=action,
                rewards=reward,
                dones=done,
                infos=info,
            )

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            # filter dead agents
            if multiagent:
                next_obs = {
                    agent_id: obs
                    for agent_id, obs in next_obs.items()
                    if not done[agent_id]
                }

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            step += 1
            obs = next_obs
        logger.info(
            "\nEpisode #{}: steps: {} reward: {}".format(episode, step, reward_total)
        )
        if done:
            episode += 1
    metrics_handler.write_to_csv(csv_dir=log_dir)

    if saved_transitions_dir is not 'None':
        import pickle
        with open( saved_transitions_dir + "/saved_transitions.pkl", "wb") as f:
            pickle.dump(saved_transitions, f)
