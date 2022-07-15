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
import argparse
import os

import ray

from baselines.marl_benchmark import gen_config
from baselines.marl_benchmark.metrics import basic_handler
from baselines.marl_benchmark.utils.rollout import rollout
from baselines.marl_benchmark.utils.rollout_orig import rollout as rollout_orig
from baselines.marl_benchmark.utils.rollout_indep_actors import rollout_indep_actors
from baselines.marl_benchmark.sv_models.sv_models import SV_GPModel
import torch
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser("Run evaluation")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name",
    )
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=10)
    # TODO(ming): eliminate this arg
    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--record_data_dir", default=None, type=str, help="Turn on data replay recording in selected dir"
    )
    parser.add_argument(
        "--indep_actors", default=False, action="store_true", help="Evaluate single-agent policies (ckpts specified in file)"
    )
    parser.add_argument("--checkpoint", type=str, default='None')
    parser.add_argument("--config_files", "-f", type=str, nargs="+", required=True)
    parser.add_argument("--log_dir", type=str, default="./log/results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--saved_transitions_dir", type=str, default='None')
    return parser.parse_args()


def main(
    scenario,
    checkpoint_path,
    config_files,
    log_dir,
    num_steps=1000,
    num_episodes=10,
    paradigm="decentralized",
    headless=False,
    record_data_dir=None,
    show_plots=False,
    saved_transitions_dir=None,
    indep_actors=False,
):

    ray.init(local_mode=True)
    metrics_handler = basic_handler.BasicMetricHandler()

    sv_model_path= None

    for config_file in config_files:
        config = gen_config(
            scenario=scenario,
            config_file=config_file,
            num_steps=num_steps,
            num_episodes=num_episodes,
            paradigm=paradigm,
            headless=headless,
            record_data=record_data_dir,
            mode="evaluation"
        )
        if sv_model_path:
            config['env_config']['custom_config']['model_path'] = sv_model_path
            config['env_config']['custom_config']['beta_t'] = 1.0
            config['env_config']['custom_config']['num_samples'] = 5
            config['env_config']['custom_config']['unif_grid'] = True
            config['env_config']['custom_config']['thompson'] = False

        tune_config = config["run"]["config"]
        trainer_cls = config["trainer"]
        trainer_config = {"env_config": config["env_config"]}
        if paradigm != "centralized":
            trainer_config.update({"multiagent": tune_config["multiagent"]})
        else:
            trainer_config.update({"model": tune_config["model"]})

        trainer_config.update({"create_env_on_driver": True}) # Otherwise worker=0 does not have an env and evaluation cannot start.

        trainer = trainer_cls(env=tune_config["env"], config=trainer_config)

        if checkpoint_path == 'None':
            #If ckpt is not explicitly passed as argument, get it from config file
            checkpoint_path = config["checkpoint"]

        trainer.restore(checkpoint_path)
        metrics_handler.set_log(
            algorithm=config_file.split("/")[-2], num_episodes=num_episodes
        )

        if indep_actors:
            trainers_actors = []
            for i in range(2):
                config_new = gen_config(
                    scenario=scenario + '_agent'+ str(i),
                    config_file=config_file,
                    num_steps=num_steps,
                    num_episodes=num_episodes,
                    paradigm=paradigm,
                    headless=headless,
                    record_data_dir=None,
                    mode="evaluation"
                )
                tune_config_new = config_new["run"]["config"]
                trainer_cls_new = config_new["trainer"]
                trainer_config_new = {"env_config": config_new["env_config"]}
                if paradigm != "centralized":
                   trainer_config_new.update({"multiagent": tune_config_new["multiagent"]})
                else:
                    trainer_config_new.update({"model": tune_config_new["model"]})
                #trainer_config_new.update({"create_env_on_driver": True})  # Otherwise worker=0 does not have an env and evaluation cannot start.
                trainer_new = trainer_cls_new(env=tune_config_new["env"], config=trainer_config_new)
                trainers_actors.append(trainer_new)

            ckpt_0 = "./log/results/run/lane_merge_agent0-4/DQN_FrameStack_b7d5c_00000_0_2022-02-09_15-08-12/checkpoint_000320/checkpoint-320"
            ckpt_1 = "./log/results/run/lane_merge_agent1-4/DQN_FrameStack_f425d_00000_0_2022-02-10_07-52-03/checkpoint_000570/checkpoint-570"
            trainers_actors[0].restore(ckpt_0)
            trainers_actors[1].restore(ckpt_1)


        sv_model = None
        if sv_model_path:
            train_x, train_y = torch.load(sv_model_path + '/training_data.pth')
            sv_model = SV_GPModel(train_x, train_y)
            sv_model.load_model(sv_model_path+'/model_state.pth')
        if indep_actors:
            rollout_indep_actors(trainer, trainers_actors, None, metrics_handler, num_steps, num_episodes, log_dir)
        elif trainer.config['env'] in ['FrameStack', 'H_FrameStack' ,'EarlyDone'] :
            rollout_orig(trainer, None, metrics_handler, num_steps, num_episodes, log_dir)
        else:
            rollout(trainer, None, metrics_handler, num_steps, num_episodes, log_dir, sv_model=sv_model, saved_transitions_dir=saved_transitions_dir)
        trainer.stop()

        fig_behavior, fig_rewards, rewards = metrics_handler.return_plots()
        fig_rewards.savefig(metrics_handler._csv_dir+'/fig_rewards.png')
        np.save(metrics_handler._csv_dir+'/rewards.npy', rewards)

        if show_plots:
            fig_behavior.show()
            fig_rewards.show()

    return

if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        checkpoint_path=args.checkpoint,
        config_files=args.config_files,
        num_steps=args.num_steps,
        num_episodes=args.num_runs,
        paradigm=args.paradigm,
        headless=args.headless,
        record_data_dir=args.record_data_dir,
        show_plots=args.plot,
        log_dir=args.log_dir,
        saved_transitions_dir=args.saved_transitions_dir,
        indep_actors=args.indep_actors
    )
