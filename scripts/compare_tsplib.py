import pandas as pd
import pathlib
import argparse
import random
import tsplib95

from egls import datasets
import numpy as np
from matplotlib import cm, pyplot as plt
plt.rcParams["font.family"] = "serif"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tsplib_path', type=pathlib.Path)
    parser.add_argument('opt_path', type=pathlib.Path)
    parser.add_argument('log_paths', type=pathlib.Path, nargs='+')
    parser.add_argument('--labels', type=str, nargs='+', default=[])
    parser.add_argument('--colors', type=str, nargs='+', default=cm.tab10.colors)
    parser.add_argument('expt_name', type=str)
    # args = parser.parse_args([
    #     '../data/tsplib/test.txt',
    #     '../data/tsplib/test_attention_tsp_opt.pkl',
    #     '../final_results/hpc/attention_tsp/iters/test_attention_tsp/test_attention_tsp-tsp_50_epoch-99-sample1280-t1-0-290_profile_processed.pkl',
    #     '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp50_tsplib_processed.pkl',
    #     '../final_results/hpc/learning-2opt-drl/iters/policy-TSP50-epoch-268/TSP100_learn_2opt_tsplib.pkl',
    #     '../final_results/hpc/tsp50/tsplib_regret_pred_10.0_20_10_True_Nov22_23-26-31_323a50798aa742b6966f16ea57936794.pkl',
    #     'test',
    #     '--labels',
    #     'Kool et al.',
    #     'Joshi et al.',
    #     'O. da Costa et al.',
    #     'Ours'
    # ])
    args = parser.parse_args([
        '../data/tsplib/test.txt',
        '../data/tsplib/test_attention_tsp_opt.pkl',
        '../final_results/hpc/attention_tsp/iters/test_attention_tsp/test_attention_tsp-tsp_100_epoch-99-sample1280-t1-0-290_profile_processed.pkl',
        '../final_results/hpc/graph-convnet-tsp/beamsearch/tsp100_tsplib_processed.pkl',
        '../final_results/hpc/learning-2opt-drl/tsplib/TSP100_learn_2opt_tsplib.pkl',
        '../final_results/hpc/tsp100/tsplib_regret_pred_10.0_20_10_True_Nov17_18-17-56_e730827179d04df6a640ac3caea64fd3.pkl',
        'test',
        '--labels',
        'Kool et al.',
        'Joshi et al.',
        'O. da Costa et al.',
        'Ours'
    ])
    opt = pd.read_pickle(args.opt_path)

    ds = datasets.TSPLIBDataset(args.tsplib_path)
    instances = []
    for instance_i, instance_name in enumerate(ds.instances):
        nice_name = pathlib.Path(instance_name).stem
        instance_path = ds.root_dir / (nice_name + '.tsp')
        instance = tsplib95.load(instance_path)
        instances.append((instance_name, instance_i, instance.dimension, nice_name))
    instances_df = pd.DataFrame(instances, columns=['name', 'idx', 'dimension', 'nice_name'])
    instances_df = instances_df.join(opt.set_index('instance'), on='idx')

    # instance_to_display = random.choice(instances_df.index)
    # print(instance_to_display)
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Computation time (s)')
    # ax.set_ylabel('Optimality gap (%)')
    # ax.set_title(instances_df.loc[instance_to_display, 'nice_name'])

    if len(args.labels) == 0:
        args.labels = args.log_paths

    ps = []
    results = []
    for log_path, log_label in zip(args.log_paths, args.labels):
        print(log_path)
        df = pd.read_pickle(log_path)

        # if 'regret_pred' in str(log_path) or 'learning-2opt-drl' in str(log_path): # ours
        #     df = df.join(instances_df.set_index('idx').drop('opt_cost', axis=1), on='instance')
        # else:
        #     df = df.join(instances_df.set_index('idx'), on='instance')

        df['cost'] = df['cost'].replace(0, np.nan)
        df['best_cost'] = df.groupby(['instance', 'run'])['cost'].cummin()
        df['gap'] = (df['best_cost'] / df['opt_cost'] - 1) * 100
        df['dt'] = df['time'] - df.groupby(['instance', 'run'])['time'].transform('min')
        # print(df['dt'].max())
        # df = df[df['dt'] <= 10.0]
        # print(df['dt'].max())
        # print(df['gap'].min())
        best = df.groupby(['instance', 'run']).apply(lambda g: g.loc[g['dt'].idxmax(), ['gap', 'dt']]).reset_index()
        print(f"{best['gap'].mean():.3f}±{best['gap'].std():.3f}")
        print(f"{best['dt'].mean():.3f}±{best['dt'].std():.3f}")
        stats = best.groupby('instance')[['gap', 'dt']].agg(['mean', 'std'])
        stats.loc[stats['gap', 'mean'] < 0, 'gap'] = 0

        for instance_i, row in stats.iterrows():
            results.append((log_label, instance_i,'Time (s)',f"{row.loc['dt', 'mean']:.3f}±{row.loc['dt', 'std']:.3f}"))
            results.append((log_label, instance_i, 'Gap (%)',
                            f"{row.loc['gap', 'mean']:.3f}±{row.loc['gap', 'std']:.3f}"))
        # stats = instances_df.join(stats, on='idx')

        # p = stats.loc[instance_to_display][('dt', 'mean')],stats.loc[instance_to_display][('gap', 'mean')]
        # ax.scatter(*p)
        # # https://github.com/Phlya/adjustText/wiki
        # ax.annotate(f'{log_label} ({p[0]:.2f},{p[1]:.2f})', p)
        # ps.append(p)

    results_df = pd.DataFrame(results, columns=['Method', 'instance_i', 'Metric', 'Value'])
    results_df = results_df.join(instances_df.set_index('idx'), on='instance_i')
    results_table = results_df.pivot(values='Value', index='nice_name', columns=['Method', 'Metric'])
    results_table = results_table.reindex(instances_df.sort_values(['dimension', 'nice_name'])['nice_name'])

    for name, row in results_table.iterrows():
        methods = row.index.levels[0]

    results_table.to_latex('tsplib_50.tex')
    # ps_df = pd.DataFrame(ps, columns=['dt', 'gap'])
    # dumb algorithm
    # min_dt = min(ps, key=lambda x:x[0])
    # min_gap = min(ps, key=lambda x:x[1])
    # pareto_points = []
    # for p in ps:
    #     if (p[1] <= min_dt[1] and p[0] <= min_gap[0]):
    #         pareto_points.append(p)
    # x, y = zip(*sorted(pareto_points, key=lambda x: x[0]))
    # ax.plot(x, y, linestyle='--', c='k', zorder=0)
    #
    # fig.tight_layout()
    # plt.show()