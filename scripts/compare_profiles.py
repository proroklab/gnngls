import pandas as pd
import pathlib
import argparse
import numpy as np
from matplotlib import cm, pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_paths', type=pathlib.Path, nargs='+')
    parser.add_argument('--labels', type=str, nargs='+', default=[])
    parser.add_argument('expt_name', type=str)
    args = parser.parse_args()

    time_limit = 10
    ts = np.linspace(0, time_limit, 1000, endpoint=True)

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    plt.grid(which='minor', axis='y')
    plt.grid(which='major', axis='x')
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    plt.grid()

    if len(args.labels) == 0:
        args.labels = list(map(str, args.log_paths))
    assert len(args.labels) == len(args.log_paths)

    rows = []
    for log_path, log_label in zip(args.log_paths, args.labels):
        df = pd.read_pickle(log_path)
        df = df[df['instance'] != df.iloc[0]['instance']] # oops :)

        df['cost'] = df['cost'].replace(0, np.nan)
        df['best_cost'] = df.groupby('instance')['cost'].cummin()
        df['gap'] = (df['best_cost']/df['opt_cost'] - 1)*100
        df['dt'] = df['time'] - df.groupby('instance')['time'].transform('min')

        gaps = []
        for instance_i, instance in df.groupby('instance'):
            gap = np.interp(ts, instance['dt'], instance['gap'])
            gaps.append(gap)
        gaps = np.vstack(gaps)

        mask = np.isnan(gaps).sum(axis=0) > 500

        median = np.nanpercentile(gaps, 50, axis=0)
        mean = np.nanmean(gaps, axis=0)
        mean[mask] = np.nan
        percentile_20 = np.nanpercentile(gaps, 25, axis=0)
        percentile_80 = np.nanpercentile(gaps, 75, axis=0)
        opt_solved = ((gaps < 0) | np.isclose(0, gaps)).sum(axis=0)/gaps.shape[0]*100
        iters = df.value_counts('instance').mean()

        rows.append((log_label, mean[-1], median[-1], percentile_20[-1], percentile_80[-1], opt_solved[-1]))

        # ax1.semilogy(ts, mean, label=log_label)
        ax1.semilogy(ts, median, label=log_label)
        ax1.fill_between(ts, percentile_20, percentile_80, alpha=0.2)
        ax1.set_xlim([0, time_limit])
        ax1.set_ylim([1e-7, 1e1])
        ax1.set_xlabel('Computation time (s)')
        ax1.set_ylabel('Gap to optimal solution (%)')
        ax1.legend(loc='upper right')
        fig1.tight_layout()
        fig1.savefig(args.expt_name + '_gap.png')

        ax2.plot(ts, opt_solved, label=log_label)
        ax2.set_xlim([0, time_limit])
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('Optimal solutions found (%)')
        ax2.set_xlabel('Computation time (s)')
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(args.expt_name + '_opt.png')

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows, columns=['Method', 'Mean', 'Median', '25th percentile','75th percentile', 'Optimal solutions (%)'])
    df['IQR'] = df['75th percentile'] - df['25th percentile']
    df[['Method', 'Median', 'IQR', 'Optimal solutions (%)']].to_latex(args.expt_name + '.tex', float_format="{:0.4f}".format, index=False)
