import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

BASE_LOGDIR = 'lunar_logs'
COLORS = ['r', 'b', 'g', 'k', 'y']
algos = os.listdir(BASE_LOGDIR)
seeds = ['1', '2', '3', '4', '5']
eval_logs = dict()
for a in algos:
    for s in seeds:
        eval_f = os.path.join(BASE_LOGDIR, a, s, 'training_logs')
        f = open(eval_f, 'r')
        lines = f.readlines()
        f.close()
        identifier = '_'.join([a, s])
        eval_logs[identifier] = np.empty((len(lines), 1))
        for i in range(len(lines)):
            l = lines[i]
            start_ind = int(l.rfind('Total cost') + 12)
            end_ind = int(l.rfind('Episode return') - 3)
            eval_logs[identifier][i, 0] = float(l[start_ind:end_ind])

with open(os.path.join('plots', 'summary_training_cost.pkl'), 'wb') as f:
    pickle.dump(eval_logs, f)

eval_means = dict()
eval_std = dict()
for a in algos:
    f_a = np.hstack([eval_logs['_'.join([a, s])] for s in seeds])
    eval_means[a] = np.mean(f_a, axis=-1)
    eval_std[a] = np.std(f_a, axis=-1)

plt.figure()
for i in range(len(algos)):
    a = algos[i]
    plt.plot(range(len(eval_means[a])), eval_means[a], c=COLORS[i], label=a)
    plt.fill_between(range(len(eval_std[a])), eval_means[a]-eval_std[a], eval_means[a]+eval_std[a], color=COLORS[i], alpha=0.2)
    # plt.ylim([-750, 200])
    plt.legend()
    plt.title('LunarLander-v2 (continuous)')

plt.savefig('plots/lunar_training_cost.png')
plt.savefig('plots/lunar_training_cost.pdf')
