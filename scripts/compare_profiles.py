import pandas as pd
import pathlib
import argparse
import numpy as np
from matplotlib import cm, pyplot as plt
plt.rcParams["font.family"] = "serif"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_paths', type=pathlib.Path, nargs='+')
    parser.add_argument('--labels', type=str, nargs='+', default=[])
    parser.add_argument('--colors', type=str, nargs='+', default=cm.tab10.colors)
    parser.add_argument('expt_name', type=str)
    args = parser.parse_args()

    time_limit = 10
    ts = np.linspace(0, time_limit, 1000, endpoint=True)

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    plt.grid(which='minor', axis='y')
    plt.grid(which='major', axis='x')
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    plt.grid()

    if args.expt_name == 'tsp20':
        axins = ax2.inset_axes([0.5, 0.03, 0.47, 0.47])
        # sub region of the original image
        axins.xaxis.tick_top()
        axins.set_xlim(9, 10)
        axins.set_ylim(90, 100.2)
        axins.set_xticks([9, 10])
        axins.set_yticks([90, 95, 100])
        # axins.set_xticklabels('')
        # axins.set_yticklabels('')
        _, zoom_connectors = ax2.indicate_inset_zoom(axins, edgecolor="black", label=None)
        zoom_connectors[0].set_visible(False)
        zoom_connectors[1].set_visible(True)
        zoom_connectors[2].set_visible(True)
        zoom_connectors[3].set_visible(False)

        # axins1 = ax1.inset_axes([0.5, 0.03, 0.47, 0.47])
        # # sub region of the original image
        # axins1.xaxis.tick_top()
        # axins1.set_xlim(-0.001, 1)
        # axins1.set_ylim(1e-7, 1e-6)
        # axins1.set_xticks([0, 1])
        # # axins1.set_yticks([1e-7, 1e-6])
        # # axins1.set_xticklabels('')
        # # axins1.set_yticklabels('')
        # _, zoom_connectors = ax1.indicate_inset_zoom(axins1, edgecolor="black", label=None)
        # zoom_connectors[0].set_visible(False)
        # zoom_connectors[1].set_visible(True)
        # zoom_connectors[2].set_visible(True)
        # zoom_connectors[3].set_visible(False)

    elif '100' in args.expt_name:
        axins = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
        # sub region of the original image
        axins.set_xlim(9, 10)
        axins.set_ylim(-0.2, 10)
        axins.set_xticks([9, 10])
        axins.set_yticks([0, 5, 10])
        # axins.set_xticklabels('')
        # axins.set_yticklabels('')
        _, zoom_connectors = ax2.indicate_inset_zoom(axins, edgecolor="black", label=None)
        zoom_connectors[0].set_visible(True)
        zoom_connectors[1].set_visible(False)
        zoom_connectors[2].set_visible(False)
        zoom_connectors[3].set_visible(True)

    if len(args.labels) == 0:
        args.labels = list(map(str, args.log_paths))
    assert len(args.labels) == len(args.log_paths)

    rows = []
    for log_path, log_label, log_color in zip(args.log_paths, args.labels, args.colors):
        print(log_label)
        df = pd.read_pickle(log_path)
        df = df[df['instance'] != df.iloc[0]['instance']] # oops :)

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

        mask = np.isnan(gaps).sum(axis=0) > 0

        median = np.nanpercentile(gaps, 50, axis=0)
        mean = np.nanmean(gaps, axis=0)
        mean[mask] = np.nan
        percentile_20 = np.nanpercentile(gaps, 25, axis=0)
        percentile_80 = np.nanpercentile(gaps, 75, axis=0)
        # opt_solved = (np.isclose(0, gaps, rtol=0, atol=1e-7)).sum(axis=0)/gaps.shape[0]*100
        print(np.nanmin(gaps))
        print(gaps.shape[0])
        opt_solved = ((gaps < 0) | np.isclose(0, gaps, rtol=0, atol=1e-8)).sum(axis=0)/gaps.shape[0]*100
        iters = df.value_counts('instance').mean()

        rows.append((log_label, mean[-1], median[-1], percentile_20[-1], percentile_80[-1], opt_solved[-1]))

        # print(median)
        # ax1.semilogy(ts, mean, label=log_label)
        ax1.semilogy(ts, median, label=log_label, color=log_color)
        ax1.fill_between(ts, percentile_20, percentile_80, alpha=0.2, color=log_color)
        ax1.set_xlim([-0.05, time_limit])
        if '100' in args.expt_name:
            ax1.set_ylim([1e-1, 1000])
        else:
            ax1.set_ylim([1e-7, 100])
        ax1.set_xlabel('Computation time (s)')
        ax1.set_ylabel('Gap to optimal solution (%)')
        ax1.legend(loc='best')
        fig1.tight_layout()
        fig1.savefig(args.expt_name + '_gap.pdf', bbox_inches='tight')

        ax2.plot(ts, opt_solved, label=log_label, ls='-', color=log_color)
        ax2.set_xlim([0, time_limit])
        ax2.set_ylim([-1, 101])
        ax2.set_ylabel('Optimal solutions found (%)')
        ax2.set_xlabel('Computation time (s)')
        ax2.legend(loc='upper left')

        # inset axes....
        # if '100' in args.expt_name or 'tsp20' == args.expt_name:
        axins.plot(ts, opt_solved, label=log_label, ls='-', color=log_color)

            # axins1.semilogy(ts, median, label=log_label)
            # axins1.fill_between(ts, percentile_20, percentile_80, alpha=0.2)

        fig2.tight_layout()
        fig2.savefig(args.expt_name + '_opt.pdf', bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows, columns=['Method', 'Mean', 'Median', '25th percentile','75th percentile', 'Optimal solutions (%)'])
    df['IQR'] = df['75th percentile'] - df['25th percentile']
    df[['Method', 'Median', 'IQR', 'Optimal solutions (%)']].to_latex(args.expt_name + '.tex', float_format="{:0.4f}".format, index=False)
