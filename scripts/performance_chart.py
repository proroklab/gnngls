import pandas as pd
import pathlib
import argparse
import random
import tsplib95
from adjustText import adjust_text

from egls import datasets
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
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp20.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP20-epoch-189/TSP20_learn_2opt_tsp20_test.pkl',
    #     '../final_results/hpc/tsp20/tsp20_test_regret_pred_10.0_20_True_Sep17_19-57-22_9485d12d27164ed3a48b0f9e1291e2d1.pkl',
    #     '../final_results/hpc/kgls/tsp20_test_weight_width_width_and_weight_10.0_20_False_Sep17_20-02-48_904aaf67de11409caf45f85d18a558f3.pkl',
    #     '../final_results/hpc/tsp20_test_concorde.pkl',
    #     '../final_results/hpc/tsp20_test_lkh.pkl',
    #     '../final_results/hpc/tsp20/tsp20_test_weight_0_30_1_False_nn_True_Nov19_11-43-08_52cfb619979742798d8c5b6414fc7bcb.pkl',
    #     '../final_results/hpc/tsp20/tsp20_test_weight_0_30_1_False_fi_True_Nov19_11-45-19_c45114f6d1f44902bf7fa9586432b0ef.pkl',
    #     '../final_results/hpc/tsp20/tsp20_test_weight_0.0_30_1_False_nn_False_Nov19_11-48-12_53339283456f4ea587464f23a5f487c3.pkl',
    #     'tsp20_perf',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Arnold et al.',
    #     'Concorde',
    #     'LKH-3',
    #     'Nearest Neighbor',
    #     'Farthest Insertion',
    #     'Local Search',
    # ])
    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/iters/test_attention_tsp50/test_attention_tsp50-tsp_50_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp50.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP50-epoch-268/TSP50_learn_2opt_tsp50_test.pkl',
    #     '../final_results/hpc/tsp50/tsp50_test_regret_pred_10.0_20_True_Sep17_14-17-51_3000ecba815c4878a58abf50ab86d453.pkl',
    #     '../final_results/hpc/kgls/tsp50_test_weight_width_width_and_weight_10.0_20_False_Sep17_20-26-24_95ecde436d0845ab9ad3ed70e196ffef.pkl',
    #     '../final_results/hpc/tsp50_test_concorde.pkl',
    #     '../final_results/hpc/tsp50_test_lkh.pkl',
    #     '../final_results/hpc/tsp50/tsp50_test_weight_0_30_1_False_nn_True_Nov19_12-12-55_bd4400e7217f4ee98c0b946a7c5858bd.pkl',
    #     '../final_results/hpc/tsp50/tsp50_test_weight_0_30_1_False_fi_True_Nov19_12-17-32_1b7832c537094f81a6f6e3eb7203678d.pkl',
    #     '../final_results/hpc/tsp50/tsp50_test_weight_0.0_30_1_False_nn_False_Nov19_11-54-02_2ac54d33804347968a609432df39da6b.pkl',
    #     'tsp50_perf',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Arnold et al.',
    #     'Concorde',
    #     'LKH-3',
    #     'Nearest Neighbor',
    #     'Farthest Insertion',
    #     'Local Search',
    # ])
    # args = parser.parse_args([
    #     '../final_results/hpc/attention_tsp/iters/test_attention_tsp100/test_attention_tsp100-tsp_100_epoch-99-sample1280-t1-0-1000_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp100.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP100-epoch-262/TSP100_learn_2opt_tsp100_test.pkl',
    #     '../final_results/hpc/tsp100/tsp100_test_regret_pred_10.0_20_True_Sep17_14-20-05_d36c0f735923482f97d1b402a6c77f44.pkl',
    #     '../final_results/hpc/kgls/tsp100_test_weight_width_width_and_weight_10.0_20_False_Sep17_20-40-05_1c235f030d654370a899ca013aab1460.pkl',
    #     '../final_results/hpc/tsp100_test_concorde.pkl',
    #     '../final_results/hpc/tsp100_test_lkh.pkl',
    #     '../final_results/hpc/tsp100/tsp100_test_weight_0_30_1_False_nn_True_Nov19_12-28-46_02ef27e2b367426f8e5d2fe698633695.pkl',
    #     '../final_results/hpc/tsp100/tsp100_test_weight_0_30_1_False_fi_True_Nov19_12-49-05_77cb3ca965394095b899befba9c9ca59.pkl',
    #     '../final_results/hpc/tsp100/tsp100_test_weight_0.0_30_1_False_nn_False_Nov19_12-11-46_9744eb445e11458ea30b96c807b7f193.pkl',
    #     'tsp100_perf',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours',
    #     'Arnold et al.',
    #     'Concorde',
    #     'LKH-3',
    #     'Nearest Neighbor',
    #     'Farthest Insertion',
    #     'Local Search',
    # ])

    min_y = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    # fig.tight_layout()
    for ax in axes:
        ax.set_xlabel('Mean computation time per instance (s)')
        ax.set_ylabel('Mean optimality gap (%)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(which='major', axis='y')
        # ax.set_ylim([min_y, None])
    # plt.suptitle('TSP50')

    if len(args.labels) == 0:
        args.labels = args.log_paths

    ps = []
    results = []
    annotations = {}
    annotations[axes[0]] = []
    annotations[axes[1]] = []

    for log_path, log_label, log_color in zip(args.log_paths, args.labels, args.colors):
        print(log_path)
        if 'Kool' in log_label or 'Ours' in log_label or 'da Costa' in log_label or 'Joshi' in log_label:
            ax = axes[1]
        else:
            ax = axes[0]
        df = pd.read_pickle(log_path)

        # if 'attention_tsp' in str(log_path): #or 'graph-convnet-tsp' in str(log_path):
        #     df = df.join(opt.set_index('instance'), on='instance')

        df['cost'] = df['cost'].replace(0, np.nan)
        df['best_cost'] = df.groupby('instance')['cost'].cummin()
        df['gap'] = (df['best_cost'] / df['opt_cost'] - 1) * 100
        df['dt'] = df['time'] - df.groupby('instance')['time'].transform('min')

        # if 'kgls' in str(log_path) or 'regret_pred' in str(log_path):
        #     df = df[df['dt'] <= 10]

        print(df['gap'].min())
        best = df.groupby('instance').apply(lambda g: g.loc[g['dt'].idxmax(), ['gap', 'dt']]).reset_index()
        stats = best[['gap', 'dt']].agg(['mean', 'std'])

        p = stats.loc['mean', 'dt'], stats.loc['mean', 'gap']
        if p[1] < min_y:
            ax.annotate(log_label, (p[0], min_y), xytext=(p[0], min_y*2.5), ha="center", arrowprops={'arrowstyle': '->'})
            # ax.arrow(p[0], min_y + 1e-4, 0, -1e-4,
            #          length_includes_head=True,
            #          head_length=1e-5, head_width=1e-2, width=1e-4)
            # t = ax.text(p[0], min_y, f'{log_label}') # ({p[0]:.2f},{p[1]:.2f})')
        else:
            ax.scatter(*p, c=log_color)
            # https://github.com/Phlya/adjustText/wiki
            t = ax.text(*p, f'{log_label}') # ({p[0]:.2f},{p[1]:.2f})')
            annotations[ax].append(t)
        ps.append((log_label, *p))
        results.append((log_label,
                        f"{stats.loc['mean', 'dt']:.3f}±{stats.loc['std', 'dt']:.3f}",
                        f"{stats.loc['mean', 'gap']:.3f}±{stats.loc['std', 'gap']:.3f}"
                        ))



    # ps_df = pd.DataFrame(ps, columns=['dt', 'gap'])
    # dumb algorithm
    min_dt = min(ps, key=lambda x:x[1])
    min_gap = min(ps, key=lambda x:x[2])
    m = (min_gap[2] - min_dt[2])/(min_gap[1] - min_dt[1])

    # pareto_points = []
    for p in ps:
        gap_proj = m*p[1] + min_dt[2]
        if gap_proj > p[2]:
            print(p, gap_proj)
        else:
            print('no', p, gap_proj)
        # if (p[1] <= min_dt[1] and p[0] <= min_gap[0]):
        #     pareto_points.append(p)
    # x, y = zip(*sorted(pareto_points, key=lambda x: x[1]))
    # axes[0].plot(x, y, linestyle='--', c='k', zorder=0)


    # ax.set_xlim([, None])
    axes[0].set_xlim(None, 35)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[0].set_ylim(min_y, None)
    axes[1].set_ylim(axes[0].get_ylim())
    # axes[0].set_title('Non-Learning Approaches')
    # axes[1].set_title('Learning Approaches')
    # fig.tight_layout()
    # TSP20
    annotations[axes[0]][0].set_y(10)
    annotations[axes[0]][0].set_x(5e-4)
    annotations[axes[0]][1].set_y(3)
    annotations[axes[0]][1].set_x(1e-3)
    annotations[axes[0]][2].set_y(1.5)
    annotations[axes[0]][2].set_x(1e-2)
    # TSP50
    # annotations[axes[0]][0].set_x(4.5)
    # annotations[axes[0]][2].set_y(16.5)
    # TSP100
    # annotations[axes[0]][0].set_x(6)
    # annotations[axes[0]][0].set_y(0.7)
    # annotations[axes[0]][2].set_y(17)
    # adjust_text(annotations[axes[0]]) # arrowprops=dict(arrowstyle='->', color='k'))
    adjust_text(annotations[axes[1]])

    fig.savefig(args.expt_name + '.pdf', bbox_inches='tight')
    plt.show()
    results_df = pd.DataFrame(results, columns=['Method', 'Computation time (s)', 'Optimality gap (%)'])
    results_df.to_latex(args.expt_name + '.tex', index=False)