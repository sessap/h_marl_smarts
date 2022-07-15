import argparse
import os
import torch
import ray
import shutil
import pandas
import numpy as np

from baselines.marl_benchmark import run, evaluate, gen_config
from baselines.marl_benchmark.sv_models import sv_models

def main(
    scenario,
    config_file,
    log_dir,
    restore_path=None,
    num_workers=1,
    horizon=1000,
    paradigm="decentralized",
    headless=True,
    cluster=False,
    learning_iters=10,
    exp_name=0,
    beta=1.0,
    thompson=False
):
    t_init = 0 # Only support '0' for now

    config = gen_config(
        scenario=scenario, config_file=config_file, paradigm=paradigm, headless=headless
    )

    config['env_config']['custom_config']['beta_t'] = beta
    config['env_config']['custom_config']['num_samples'] = 5
    config['env_config']['custom_config']['unif_grid'] = True
    config['env_config']['custom_config']['thompson_sampling'] = thompson


    config['run']['stop']['training_iteration'] = 0
    iters_per_run = 50
    config['run']['checkpoint_freq'] = 5

    scenario_dir = scenario.split('/')[-1] + '-2'
    T = learning_iters # number of learning rounds
    config_file_eval = ['marl_benchmark/agents/dqn/baseline-lane-control_eval.yaml']

    if t_init == 0:
        # Initialize SV model with initial (5 samples) data
        sv_model_path = os.path.abspath('sv_models/sv_init_model')
        ckpt_path = None
    else:
        logdir_t_prev =log_dir + '/exp_' + str(exp_name) + '/round_' + str(t_init-1) + '/'
        sv_model_path = logdir_t_prev + 'new_sv_model'

    for t in np.arange(0,T):
        logdir_t =log_dir + '/exp_' + str(exp_name) + '/round_' + str(t) + '/'

        # Update hallucinated model path
        config['env_config']['custom_config']['model_path'] = sv_model_path
        restore_path = ckpt_path
        config['run']['stop']['training_iteration'] += iters_per_run
        # Train policies
        run.main(
                    scenario=scenario,
                    config_file='',
                    log_dir=logdir_t,
                    restore_path=restore_path,
                    num_workers=num_workers,
                    horizon=horizon,
                    paradigm=paradigm,
                    headless=headless,
                    cluster=cluster,
                    config=config
        )
        ray.shutdown()


        if os.path.isfile(logdir_t + 'saved_transitions.pkl'):
            save_dir = 'None'
        else:
            save_dir = logdir_t

        # Evaluate last 4 checkpoints
        for idx in [config['run']['stop']['training_iteration']-5*i for i in reversed(range(4))]:
            idx_max1 = str(idx)
            if int(idx_max1) < 10:
                idx_max2 = '000' + idx_max1
            elif 9 < int(idx_max1) < 100:
                idx_max2 = '00' + idx_max1
            elif 99 < int(idx_max1) < 1000:
                idx_max2 = '0' + idx_max1
            else:
                idx_max2 = idx_max1
            ckpt_path = logdir_t + 'run/' + scenario_dir + '/' + dir + '/checkpoint_00' + idx_max2 + '/checkpoint-' + idx_max1

            record_data_dir = None
            if t in [0,1,2,3,4,9,19,24,29] and idx == config['run']['stop']['training_iteration']:
                record_data_dir = logdir_t + 'data_replay/'
            evaluate.main(
                        scenario,
                        ckpt_path,
                        config_file_eval,
                        logdir_t,
                        num_steps=horizon,
                        num_episodes=10,
                        paradigm="decentralized",
                        headless=headless,
                        show_plots=False,
                        saved_transitions_dir=save_dir,
                        record_data_dir=record_data_dir)
            ray.shutdown()

        # Load, update and save GP model
        data_x, data_y = sv_models.data_from_transitions(logdir_t + 'saved_transitions.pkl')
        old_data_x, old_data_y = torch.load(sv_model_path+'/training_data.pth')
        inputs = torch.cat([old_data_x, data_x])
        targets = torch.cat([old_data_y, data_y])
        SV_model = sv_models.SV_GPModel(inputs, targets)
        SV_model.train()
        os.mkdir(logdir_t + 'new_sv_model/')
        SV_model.save_model(logdir_t + 'new_sv_model/model_state.pth')
        torch.save([inputs, targets], logdir_t + 'new_sv_model/training_data.pth')

        # Update model path (for next iteration)
        sv_model_path = logdir_t + 'new_sv_model'
        del SV_model, inputs, targets


def parse_args():
    parser = argparse.ArgumentParser("Benchmark learning")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name",
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--headless", default=True, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--log_dir",
        default="./log/results",
        type=str,
        help="Path to store RLlib log and checkpoints, default is ./log/results",
    )
    parser.add_argument("--config_file", "-f", type=str, required=True)
    parser.add_argument("--restore_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1, help="RLlib num workers")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument(
        "--horizon", type=int, default=150, help="Horizon for a episode"
    )
    parser.add_argument(
        "--learning_iters", type=int, default=30, help="Learning iterations"
    )
    parser.add_argument(
        "--exp_name", type=str, default='0', help="Experiment id (to save results)"
    )
    parser.add_argument(
        "--beta", type=float, default=0, help="Exploration factor beta_t", required=False
    )
    parser.add_argument(
        "--thompson", default=False, action="store_true", help="Use Thompson sampling as exploration strategy"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_file=args.config_file,
        log_dir=args.log_dir,
        restore_path=args.restore_path,
        num_workers=args.num_workers,
        horizon=args.horizon,
        paradigm=args.paradigm,
        headless=args.headless,
        cluster=args.cluster,
        learning_iters=args.learning_iters,
        exp_name=args.exp_name,
        beta=args.beta,
        thompson=args.thompson
    )
