import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
from baselines.marl_benchmark.metrics.basic_handler import BasicMetricHandler
exp_list = ['exp_lane_merge_known_model', 'exp_lane_merge_beta_0', 'exp_lane_merge_beta_1', 'exp_lane_merge_thompson']
exp_list = ['exp_intersection_known_model', 'exp_intersection_beta_0', 'exp_intersection_beta_1', 'exp_intersection_thompson']
legend = [ 'known model', 'pred. mean', 'H-MARL', 'TS']
assert len(legend) == len(exp_list), "legend should be consistent with exp list"

plt.style.use("seaborn-colorblind") #ggplot

figs = []
axis = []
for i in range(2):
    size = (6,3.5)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    figs.append(fig)
    axis.append(ax)

for exp_path in exp_list:
    exp_dir = './log/results/' + exp_path
    if len(exp_list) == 1:
        savedir = exp_dir
    elif 'intersection' in exp_dir:
        savedir = './log/results/figs/intersection'
    elif 'lane_merge' in exp_dir:
        savedir = './log/results/figs/lane_merge'
    T = len(os.listdir(exp_dir))

    rewards_list = []
    times_to_completion_list = []
    collisions_list = []
    completions_list = []
    speeds_agent_0 = []
    accels_agent_0 = []
    speeds_agent_1 = []
    accels_agent_1 = []
    for t in range(T):
        logdir_t = exp_dir + '/round_' + str(t) + '/'
        if 1:
            rewards_t = []
            times_to_completion_t = []
            completions_t = []
            for i in np.arange(0,4):
                run_dir = sorted(os.listdir(logdir_t))[i]
                rewards = np.load(logdir_t + run_dir + '/rewards.npy')
                rewards_t.extend(rewards)

                metrics_handler = BasicMetricHandler()
                agent_metrics = metrics_handler.compute(logdir_t+run_dir, return_metrics=True)
                agent_ids = list(agent_metrics.keys())
                times_to_completion = np.array([np.mean(agent_metrics[i]['Episodes length']) for i in agent_ids]).T
                completions = np.array([agent_metrics[i]['Completion Rates'] for i in agent_ids]).T
                times_to_completion_t.append(np.mean(times_to_completion))
                completions_t.extend(completions)
            times_to_completion_list.append(times_to_completion_t)
            completions_list.append(completions_t)
            rewards_list.append(rewards_t)
        else:
            break
        T = len(rewards_list)

    rewards_list = [np.array(r) for r in rewards_list] # list of (N_episodes x N_players) reward matrices
    times_to_completion_list = [np.array(c) for c in times_to_completion_list] # list of (N_episodes x N_players) reward matrices
    completions_list = [np.array(c) for c in completions_list] # list of (N_episodes x N_players) reward matrices

    def plot_avg_mean_and_stds(ax, list_to_plot):
        cumul_list = np.cumsum(np.array(list_to_plot), axis=0)
        for i in range(len(cumul_list[0,:])):
            cumul_list[:, i] = np.divide(cumul_list[:, i], np.arange(1, T + 1))
        means = np.mean(cumul_list,axis=1)
        stds1 = np.percentile(cumul_list, 30, axis=1)
        stds2 = np.percentile(cumul_list, 70, axis=1)

        if legend[exp_list.index(exp_path)] == 'known model':
            ax.plot(np.arange(1, T + 1), [means[-1]]*T, ':', color='dimgray', linewidth=2.0, label=legend[exp_list.index(exp_path)])
        else:
            ax.plot(np.arange(1, T + 1), means, linewidth=2.0, label=legend[exp_list.index(exp_path)])
            ax.fill_between(np.arange(1, T + 1), stds1, stds2, alpha=0.15, edgecolor=None, label='_nolegend_')
        ax.grid( color = 'gray', alpha=0.2)

    avg_rewards_list = [np.mean(r, axis = 1) for r in rewards_list]
    plot_avg_mean_and_stds(axis[0], avg_rewards_list)
    axis[0].set_xlabel('round')
    axis[0].set_ylabel('avg. episodic reward')

    avg_completions_list = [np.mean(r, axis = 1) for r in completions_list]
    plot_avg_mean_and_stds(axis[1], avg_completions_list)
    axis[1].set_xlabel('round')
    axis[1].set_ylabel('avg. completion rate')

    avg_final_completions = []
    for i in range(T): #range(3)
        avg_final_completions.extend(list(avg_completions_list[-1-i]))

    avg_times_to_completion_list = [np.nanmean(r) for r in times_to_completion_list ]
    avg_final_times_to_completion = []
    for i in range(T): #range(3)
        avg_final_times_to_completion.append(avg_times_to_completion_list[-1 - i])

    print()
    print(str(legend[exp_list.index(exp_path)]))
    print()
    print(' Completion rate during learning:    ' + str(np.mean(avg_final_completions)) + ',  std. = ' + str(np.std(avg_final_completions)))
    print(' Avg. completion time: ' + str(np.nanmean(avg_final_times_to_completion))+ ',  std. = '+ str(np.std(avg_final_times_to_completion)))


    """ Comparison with model-free DQN """
    if 'known_model' in exp_dir:
        mean_modelfree_rewards =  np.array([np.mean(r) for r in avg_rewards_list])
        stds_modelfree_rewards = np.array([np.std(r) for r in avg_rewards_list])
        model_free_rewards = []
        played_episodes = [0]
        for t in range(T):
            logdir_t = exp_dir + '/round_' + str(t) + '/'
            trainer_dir = logdir_t + 'run/' +  os.listdir(logdir_t+ 'run/')[0] + '/'
            for dir in os.listdir(trainer_dir):
                if 'DQN' in dir:
                    progress = pandas.read_csv(trainer_dir + dir  + '/progress.csv',
                                               usecols=['episode_reward_mean', 'episodes_total'])
                    model_free_rewards.extend(0.5*progress['episode_reward_mean'].values) #since values are the sum of the two-agents rewards
                    played_episodes.extend(list(progress['episodes_total'].values))
                    break
        played_episodes = played_episodes[1:]

    if 'beta_1' in exp_dir:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(7, 3.5))
        means = np.array([np.mean(r) for r in avg_rewards_list])
        stds1 = means - np.array([np.std(r) for r in avg_rewards_list])  #
        stds2 = means + np.array([np.std(r) for r in avg_rewards_list])
        ax1.plot(np.arange(1, T + 1), means, '.-', linewidth=2.0, color=axis[0].get_lines()[-1].get_color())
        ax1.fill_between(np.arange(1, T + 1), stds1, stds2, color=axis[0].get_lines()[-1].get_color(), alpha=0.2,
                         label='_nolegend_')
        ax1.plot(played_episodes, model_free_rewards, '.-', color='black')
        ax2.plot(played_episodes, model_free_rewards, '.-', color='black')
        fig.supxlabel('# of interactions with the environment (played episodes)', fontsize=10)
        ax1.set_ylabel('avg. episodic reward')
        ax1.legend(['H-MARL', 'DQN (model-free)'], loc='lower left', fontsize=12)
        ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax1.set_xlim([0.5, 31])
        ax2.set_xlim([30, 17000])
        ax1.set_ylim([-125, 65])
        ax2.set_ylim([-125, 65])
        # hide the spines between ax and ax2
        ax1.spines.right.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax1.yaxis.tick_left()
        ax1.tick_params(labelright=False)  # don't put tick labels at the top
        ax2.yaxis.tick_right()

        # Now, let's turn towards the cut-out slanted lines.
        # We create line objects in axes coordinates, in which (0,0), (0,1),
        # (1,0), and (1,1) are the four corners of the axes.
        # The slanted lines themselves are markers at those locations, such that the
        # lines keep their angle and position, independent of the axes size or scale
        # Finally, we need to disable clipping.
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
        fig.tight_layout()
        fig.savefig(savedir + '/comparison_with_modelfree.png')

for i in range(len(figs)):
    loc = 'best'
    axis[i].legend(legend, loc=loc, fontsize=14)
    figs[i].tight_layout()

figs[0].savefig(savedir+'/fig_avg_learning.png')
figs[1].savefig(savedir+'/fig_avg_completion_rates.png')

