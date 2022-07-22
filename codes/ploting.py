import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from matplotlib import colors as mcolors
from glob import glob

def avg_cost_plot(cost_history, iteration, days, epochs, title):
    """
    시간 평균 그래프
    """
    temp = 0
    check_dqn = list()
    check_greedy = list()
    for i in range(0,iteration):
        temp = temp + sum(cost_history[i*days : i*days+days])
        temp2 = (temp/(i+1))
        check_dqn.append(temp2)
    with plt.style.context('ggplot'):
        plt.figure(figsize=(15,10))
        x = range(0, int(epochs / days))
        y1 = [check_dqn[v] for v in x] #dqn
        plt.plot(x,y1,label='DQL time avgerage cost', color='r')
        plt.plot([0, x[-1]], [y1[-1], y1[-1]], "b:")
        plt.plot(x[-1], y1[-1], "bo")
        plt.text(x[-1]-0.04*x[-1], y1[-1]+0.04*y1[-1],"{}".format(y1[-1]))
        plt.title("DQL? Time Avegerage Cost / {}".format(title), fontsize=15)
        plt.xlabel('Epochs',fontsize=22)
        plt.ylabel('Cost(won)',fontsize=22)
        plt.legend()
        name_ = title.split("_")[0]
        print("{}_price : {}".format(name_, sum(cost_history[epochs-days:epochs])))
        print("{}_ave : {}".format(name_, check_dqn[int(epochs/days)-1]))
        # plt.show()
        plt.savefig("{}.png".format(title))

def charge_graph(load_data, generation_data, TOU, action_history, battery_history, path):
    idxs = np.random.choice(len(load_data), 4, replace=False)
    fig, ax = plt.subplots(2, 2, figsize=(2*7, 2*5))
    for iters, i in enumerate(idxs):
        row = iters // 2
        col = iters % 2
        x = range(0, 24)
        y1 = [v for v in load_data[i]]
        y2 = [v for v in action_history[i]]
        y3 = [v for v in TOU[0:24]]
        y4 = [v for v in battery_history[i]]
        y5 = [v for v in generation_data[i]]

        ax[row, col].plot(x, y3, label='TOU', color='gray')
        ax[row, col].fill_between(x[0:24], y3[0:24], color='lightgray')

        #plt.plot(x, y4, linewidth=3, label='HM_charge', color='Red')
        ax[row, col].plot(x, y2, linewidth=3 ,label='DQL', color='Orange')
        #plt.plot(x, y5, linewidth=3 ,label='Optimal', color='orange')

        ax[row, col].plot(x, y1,'-', label='Load', color='black')
        ax[row, col].plot(x, y5, '--',label='Generation',color='gray')
        ax[row, col].plot(x, y4,'b--', label='battery')

        ax[row, col].set(title='{} Day sample'.format(i+1),
                         ylabel='Charge', xlabel='Hour',
                        xticks=np.arange(0, 24), yticks=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]) 
        ax[row, col].legend(loc='best')
        ax[row, col].grid(True)

    fig.suptitle('_'.join(path.split("_")[1:3]),fontweight ="bold")
    fig.tight_layout()
    plt.show()

def violation_accumulation(model_list):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    ran_color = np.random.choice(list(colors.keys()), size=len(model_list))

    with plt.style.context('ggplot'):
        plt.figure(figsize=(8,6))
        for model_name, cor in zip(model_list, ran_color):
            path = glob('history/violation/' + model_name + '/*.pkl')

            violation_history = [pickle.load(open(pt, 'rb'))['violation'] for pt in path]

            min_ = np.min(violation_history, axis=0)
            max_ = np.max(violation_history, axis=0)
            mean_ = np.mean(violation_history, axis=0)

            plt.plot(min_, linewidth=1, color=cor)
            plt.plot(max_, linewidth=1, color=cor)
            plt.plot(mean_, label=f'{model_name} violation aumm', linewidth=2, color=cor, linestyle=":")
            plt.fill_between(range(len(mean_)) , min_, max_, facecolor=cor, alpha=0.3)
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("count")
        plt.title("Violation accumulation", fontsize=20)
        plt.show()
    del colors, ran_color, min_, max_, mean_
    pass