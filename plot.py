import argparse
import io
import os

import json
import argparse
import os
import os.path as osp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DIV_LINE_WIDTH = 50
exp_idx = 0
units = dict()

def read_data(d, f, timesteps=1e6, action=4):
    fn = osp.join(d, f)
    lines = io.open(fn, 'r', newline='\n', encoding='utf8').readlines()
    data = dict()
    for line in lines:
        di = json.loads(line)
        for k, v in di.items():
            if k not in data:
                data[k] = [v]
            else:
                data[k].append(v)
        if data['step'][-1] * action > timesteps:
            break
    return pd.DataFrame(data)

def get_datasets(logdir, condition=None, **kwargs):
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'eval.log' in files:
            condition1 = condition
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1
            try:
                exp_data = read_data(root, 'eval.log', timesteps=kwargs['timesteps'],
                                     action=kwargs['action_repeat'])
            except Exception as e:
                print('Could not read from %s' % os.path.join(root, 'eval.log'))
                print(e)
                exit(1)
            performance = kwargs.get('yaxis', 'eprewmean')
            xaxis = kwargs.get('xaxis', 'step')
            exp_data[xaxis] *= kwargs.get('action_repeat', 4)
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets



def get_all_dataset(logdir, legend=None, select=None, exclude=None, **kwargs):
    logdirs = []
    log2class = dict()
    clses = []
    class2lengend = dict()
    subdirs = os.listdir(logdir)
    subdirs = sorted(subdirs)
    for subdir in subdirs:
        if select is None or all(x in subdir for x in select):
            if exclude is None or all(x not in subdir for x in exclude):
                cls = '-'.join(subdir.split('-')[:3] + subdir.split('-')[5:-1])
                log2class[subdir] = cls
                if cls not in clses:
                    clses.append(cls)
                logdirs.append(subdir)
    if legend is None or len(clses) != len(legend):
        print('Subdir...\n' + '=' * DIV_LINE_WIDTH + '\n')
        for subdir in clses:
            print(subdir)
        print('\n' + '=' * DIV_LINE_WIDTH)
        if legend is not None:
            print('Legend...\n' + '=' * DIV_LINE_WIDTH + '\n')
            for l in legend:
                print(l)
            print('\n' + '=' * DIV_LINE_WIDTH)
        else:
            print('legend is None')
        exit()
    for cls, leg in zip(clses, legend):
        class2lengend[cls] = leg

    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)
    data = []
    for log in logdirs:
        leg = class2lengend[log2class[log]]
        print('\nloading data from {}  /  {}\n'.format(log, leg))
        data.extend(get_datasets(osp.join(logdir, log), leg, **kwargs))
    return data


def plot_data(data, xaxis='timesteps', value="Performance", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    fig = plt.figure(kwargs['figname'], figsize=(8, 6))
    sns.set(style="darkgrid", font_scale=2)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd',)
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(loc='best', fontsize=24 ).set_draggable(True)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axis = plt.gca()
    if not kwargs['disable_vline']:
        axis.axvline(100000, color='c', lw=1)
        axis.axvline(500000, color='r', lw=1)
    plt.tight_layout(pad=0.5)

    figdir = 'data/figs'
    os.makedirs(figdir, exist_ok=True)
    figname = osp.join(figdir, '{}.pdf'.format(kwargs['figname']))
    fig.savefig(figname, format='pdf')
    plt.show()

def make_plots(logdir, legend, xaxis, yaxis, xlabel, ylabel,
               count, smooth, select, exclude, est, **kwargs):
    data = get_all_dataset(logdir, legend, select, exclude,
                    xaxis=xaxis, yaxis=yaxis, **kwargs)
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, est)
    xlabel = xlabel if xlabel else xaxis
    ylabel = ylabel if ylabel else yaxis
    plot_data(data, xaxis=xaxis, value='Performance', condition=condition, smooth=smooth,
              estimator=estimator, xlabel=xlabel, ylabel=ylabel, **kwargs)


def changeargs(args):
    env_task = args.figname
    args.action_repeat = 4
    args.timesteps = 1e6
    if env_task in ['cartpole-swingup_sparse',]:
        args.timesteps = 3e6
    if env_task in ['cartpole-swingup']:
        args.action_repeat = 8
    if env_task in ['hopper-hop', 'pendulum-swingup']:
        args.timesteps = 4e6
    if env_task in ['finger-spin', 'walker_walk']:
        args.action_repeat = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str, )
    parser.add_argument('--legend', type=str, nargs='*', default=None)
    parser.add_argument('--xaxis', '-x', type=str, default='step', )
    parser.add_argument('--yaxis', '-y', type=str, default='mean_episode_reward', )
    parser.add_argument('--xlabel', type=str, default='Environment Steps')
    parser.add_argument('--ylabel', type=str, default='Episode Score')
    parser.add_argument('--figname', type=str, default='fig')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*', default=None)
    parser.add_argument('--exclude', nargs='*', default=None)
    parser.add_argument('--est', default='mean')
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--disable-vline', action='store_true', default=False)
    args = parser.parse_args()
    changeargs(args)
    make_plots(**vars(args))

if __name__ == "__main__":
    main()