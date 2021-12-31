import pandas as pd
import pathlib
import argparse
import numpy as np
from matplotlib import cm, pyplot as plt
plt.rcParams["font.family"] = "serif"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('opt_path', type=pathlib.Path)
    parser.add_argument('log_paths', type=pathlib.Path, nargs='+')
    parser.add_argument('--labels', type=str, nargs='+', default=[])
    parser.add_argument('--colors', type=str, nargs='+', default=cm.tab10.colors)
    parser.add_argument('expt_name', type=str)
    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/iters/test_attention_tsp20/test_attention_tsp20-tsp_20_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp20_processed.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP20-epoch-189/TSP20_learn_2opt_tsp20_test.pkl',
    #     '../final_results/hpc/tsp20/tsp20_test_regret_pred_10.0_20_True_Sep17_19-57-22_9485d12d27164ed3a48b0f9e1291e2d1.pkl',
    #     '../final_results/hpc/tsp20_test_concorde.pkl',
    #     'tsp20',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Concorde',
    # ])
    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/iters/test_attention_tsp50/test_attention_tsp50-tsp_50_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp50_processed.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP50-epoch-268/TSP50_learn_2opt_tsp50_test.pkl',
    #     '../final_results/hpc/tsp50/tsp50_test_regret_pred_10.0_20_True_Sep17_14-17-51_3000ecba815c4878a58abf50ab86d453.pkl',
    #     '../final_results/hpc/tsp50_test_concorde.pkl',
    #     'tsp50',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Concorde',
    # ])
    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/time/test_attention_tsp100/test_attention_tsp100-tsp_100_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp100_processed.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP100-epoch-262/TSP100_learn_2opt_tsp100_test.pkl',
    #     '../final_results/hpc/tsp100/tsp100_test_regret_pred_10.0_20_True_Sep17_14-20-05_d36c0f735923482f97d1b402a6c77f44.pkl',
    #     '../final_results/hpc/tsp100_test_concorde.pkl',
    #     'tsp100',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Concorde',
    # ])

    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/time/test_attention_tsp50/test_attention_tsp50-tsp_20_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp20_50_processed.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP20-epoch-189/TSP50_learn_2opt_tsp50_test.pkl',
    #     '../final_results/hpc/tsp20/tsp50_test_regret_pred_10.0_20_True_Sep18_02-43-37_c19ed34c982843be87357a31912d58cc.pkl',
    #     '../final_results/hpc/tsp50_test_concorde.pkl',
    #     'tsp20_50',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Concorde',
    # ])
    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/time/test_attention_tsp100/test_attention_tsp100-tsp_50_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp50_100_processed.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP50-epoch-268/TSP100_learn_2opt_tsp100_test.pkl',
    #     '../final_results/hpc/tsp50/tsp100_test_regret_pred_10.0_20_True_Sep18_03-15-59_20ea51c9ded44c10ac77a2ba3a18c74e.pkl',
    #     '../final_results/hpc/tsp100_test_concorde.pkl',
    #     'tsp50_100',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Concorde',
    # ])
    args = parser.parse_args([
        '../final_results/hpc/attention_tsp/time/test_attention_tsp100/test_attention_tsp100-tsp_20_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
        '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp20_100_processed.pkl',
        '../final_results/hpc/learning-2opt-drl/iters/policy-TSP20-epoch-189/TSP100_learn_2opt_tsp100_test.pkl',
        '../final_results/hpc/tsp20/tsp100_test_regret_pred_10.0_20_True_Sep18_03-16-02_12a336468ee44babbc3bdb6b4d854ccb.pkl',
        '../final_results/hpc/tsp100_test_concorde.pkl',
        'tsp20_100',
        '--labels',
        'Kool et al.',
        'Joshi et al.',
        'O. da Costa et al.',
        'Ours',
        'Concorde',
    ])

    # args = parser.parse_args([
    #     '../final_results/hpc/tsp100/tsp100_test_weight_0.0_30_1_False_nn_False_Nov19_12-11-46_9744eb445e11458ea30b96c807b7f193.pkl',
    #     '../final_results/hpc/sensitivity_analysis/Oct01_10-41-01_8a4d178003c14cacaf0a5978cd5457db/tsp100_test_regret_pred_10.0_20_True_Oct04_00-44-56_1b064f6c981340a3be8836959fbb4536.pkl',
    #     '../final_results/hpc/sensitivity_analysis/Sep30_20-49-29_eda460244e3140d9b11a76b6670b2d75/tsp100_test_regret_pred_10.0_20_True_Oct04_00-44-55_045c0a2a0c0d43d79f5deb0fa31ebe6e.pkl',
    #     'tsp20_100_sens',
    #     '--labels',
    #     'Local search only',
    #     'Edge weight only',
    #     'Top 5 features'
    # ])

    time_limit = 10
    ts = np.linspace(0, time_limit, 1000, endpoint=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    # plt.grid(which='minor', axis='y')

    # if args.expt_name == 'tsp20':
    # axins = ax[1].inset_axes([0.5, 0.05, 0.475, 0.45])
    # # axins = ax[1].inset_axes([0.725, 0.05, 0.25, 0.45])
    # # sub region of the original image
    # axins.xaxis.tick_top()
    # axins.set_xlim(9, 10)
    # axins.set_ylim(90, 100.2)
    # axins.set_xticks([9, 10])
    # axins.set_yticks([90, 95, 100])
    # # axins.set_xticklabels('')
    # # axins.set_yticklabels('')
    # _, zoom_connectors = ax[1].indicate_inset_zoom(axins, edgecolor="black", label=None)
    # zoom_connectors[0].set_visible(False)
    # zoom_connectors[1].set_visible(True)
    # zoom_connectors[2].set_visible(True)
    # zoom_connectors[3].set_visible(False)

    # axins1 = ax[0].inset_axes([0.5, 0.05, 0.475, 0.45])
    # # sub region of the original image
    # axins1.xaxis.tick_top()
    # axins1.set_xlim(-0.001, 1)
    # axins1.set_ylim(1e-7, 1e-6)
    # axins1.set_xticks([0, 1])
    # # axins1.set_yticks([1e-7, 1e-6])
    # # axins1.set_xticklabels('')
    # # axins1.set_yticklabels('')
    # _, zoom_connectors = ax[0].indicate_inset_zoom(axins1, edgecolor="black", label=None)
    # zoom_connectors[0].set_visible(False)
    # zoom_connectors[1].set_visible(True)
    # zoom_connectors[2].set_visible(True)
    # zoom_connectors[3].set_visible(False)

    # elif '100' in args.expt_name:
    # axins = ax[1].inset_axes([0.5, 0.5, 0.475, 0.45])
    # # sub region of the original image
    # axins.set_xlim(9, 10)
    # axins.set_ylim(-0.2, 10)
    # axins.set_xticks([9, 10])
    # axins.set_yticks([0, 5, 10])
    # # axins.set_xticklabels('')
    # # axins.set_yticklabels('')
    # _, zoom_connectors = ax[1].indicate_inset_zoom(axins, edgecolor="black", label=None)
    # zoom_connectors[0].set_visible(True)
    # zoom_connectors[1].set_visible(False)
    # zoom_connectors[2].set_visible(False)
    # zoom_connectors[3].set_visible(True)

    if len(args.labels) == 0:
        args.labels = list(map(str, args.log_paths))
    assert len(args.labels) == len(args.log_paths)

    rows = []
    for log_path, log_label, log_color in zip(args.log_paths, args.labels, args.colors):
        print(log_label)
        df = pd.read_pickle(log_path)
        # df = df[df['instance'] != df.iloc[0]['instance']] # oops :)

        # if 'attention_tsp' in str(log_path):
        #     opt = pd.read_pickle(args.opt_path)
        #     df = df.join(opt.set_index('instance_i'), on='instance')

        df['cost'] = df['cost'].replace(0, np.nan)
        df['best_cost'] = df.groupby('instance')['cost'].cummin()
        df['gap'] = (df['best_cost']/df['opt_cost'] - 1)*100
        df['dt'] = df['time'] - df.groupby('instance')['time'].transform('min')

        print(len(df['instance'].unique()))

        gaps = []
        for instance_i, instance in df.groupby('instance'):
            gap = np.interp(ts, instance['dt'], instance['gap'])
            gaps.append(gap)
        gaps = np.vstack(gaps)

        mask = np.isnan(gaps).sum(axis=0) > 200

        median = np.nanpercentile(gaps, 50, axis=0)
        mean = np.nanmean(gaps, axis=0)
        std = np.nanstd(gaps, axis=0)
        # if 'Joshi' in log_label:
        mean[mask] = np.nan

        percentile_20 = np.nanpercentile(gaps, 25, axis=0)
        percentile_80 = np.nanpercentile(gaps, 75, axis=0)
        # opt_solved = (np.isclose(0, gaps, rtol=0, atol=1e-7)).sum(axis=0)/gaps.shape[0]*100
        print(np.nanmin(gaps))
        print(gaps.shape[0])
        opt_solved = ((gaps < 0) | np.isclose(0, gaps, rtol=0, atol=1e-8)).sum(axis=0)/gaps.shape[0]*100
        iters = df.value_counts('instance').mean()
        # opt_solved[mask] = np.nan
        rows.append((log_label, mean[-1], std[-1], median[-1], percentile_20[-1], percentile_80[-1], opt_solved[-1]))

        # print(median)
        # ax[0].semilogy(ts, mean, label=log_label)
        if log_label == 'Concorde':
            ax[0].semilogy(ts, mean, label=log_label, color='k', ls='--')
        else:
            ax[0].semilogy(ts, mean, label=log_label, color=log_color)
        # ax[0].fill_between(ts, percentile_20, percentile_80, alpha=0.2, color=log_color)
        ax[0].set_xlim([-0.05, time_limit])
        if '100' in args.expt_name:
            # ax[0].set_ylim([1e-0, 1000])
            ax[0].set_ylim([1e-0, 100])
        elif '50' in args.expt_name:
            ax[0].set_ylim([1e-1, 100])
        else:
            ax[0].set_ylim([1e-7, 100])
        ax[0].set_xlabel('Computation time (s)')
        ax[0].set_ylabel('Mean optimality gap (%)')
        # fig.tight_layout()
        # fig.savefig(args.expt_name + '_gap.pdf', bbox_inches='tight')
        if log_label == 'Concorde':
            ax[1].plot(ts, opt_solved, label=log_label, ls='--', color='k')
            # axins.plot(ts, opt_solved, label=log_label, ls='--', color='k')
        else:
            ax[1].plot(ts, opt_solved, label=log_label, ls='-', color=log_color)
            # axins.plot(ts, opt_solved, label=log_label, ls='-', color=log_color)
        ax[1].set_xlim([0, time_limit])
        ax[1].set_ylim([-1, 101])
        ax[1].set_ylabel('Optimal solutions found (%)')
        ax[1].set_xlabel('Computation time (s)')
        # ax[1].legend(loc='upper left')

        # inset axes....
        # if '100' in args.expt_name or 'tsp20' == args.expt_name:
        # axins.plot(ts, opt_solved, label=log_label, ls='-', color=log_color)

        # axins1.semilogy(ts, median, label=log_label)
        # axins1.fill_between(ts, percentile_20, percentile_80, alpha=0.2)

        # fig2.tight_layout()

    ax[0].grid(which='major', axis='y')
    ax[1].grid(which='major', axis='y')
    fig.subplots_adjust(bottom=0.25)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0))
    fig.savefig(args.expt_name + '.pdf', bbox_inches='tight')
    # plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows, columns=['Method', 'Mean', 'Std', 'Median', '25th percentile','75th percentile', 'Optimal solutions (%)'])
    df['IQR'] = df['75th percentile'] - df['25th percentile']
    df['mean_std'] = df.apply(lambda r: f"{r['Mean']:.3f}±{r['Std']:.3f}", axis=1)
    df[['Method', 'mean_std', 'Optimal solutions (%)']].to_latex(args.expt_name + '.tex', float_format="{:0.4f}".format, index=False)
