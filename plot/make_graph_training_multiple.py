'''
TODO: INCLUDE CUMULATIVE TIME STEPS IN DATAFRAME
'''

import sys

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
from IPython.display import display

import argparse

import statistics

def get_args():

    # example: python3 make_graph_training.py --file test_data/test_ppo_actor_1.txt --title 'Test Results' --save

    parser = argparse.ArgumentParser()

    parser.add_argument('--title', dest='title', type=str, default='Training Results')
    parser.add_argument('--save', dest='save', action='store_true', help='data to figures/<title>.eps')

    args = parser.parse_args()

    return args

def extract_data(filepath):
    '''Extract the iteration and episodic return from the logging data '''
    # x is iterations so far, y1 is average episodic reward, y2 is average length
    x, y1, y2 = [], [], []

    # extract out x's and y's
    with open(filepath, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Episodic Return' in l:
                y1.append(float(l[1]))
            elif 'Average Episodic Length' in l:
                y2.append(float(l[1]))
            elif 'Iteration' in l:
                x.append(int(l[1]))

    
    return x, y1, y2

def graph_data(filepaths, args):
    ''' Plots Data '''
    
    x, y1, y2 = [], [], []
    
    for filepath in filepaths:
        x_aux, y1_aux, y2_aux = extract_data(filepath)
        x.append(x_aux)
        y1.append(y1_aux)
        y2.append(y2_aux)

    # visual check
    #print(f"x = {x}")
    #print(f"y1 = {y1}")
    #print(f"y2 = {y2}")

    ## merge into one dataset (per episode)
    # minimum episode sequence from x
    x_max = []
    for i in x:
        #print(f"i: {max(i)}")
        x_max.append(max(i))
    x_lim = min(x_max)
    #print(f"x_lim: {x_lim}")
    x_plot = []
    x_plot.extend(range(x_lim))
    #print(x_plot)
    
    # generate cut lists, then zip # not generic yet (will need a loop)
    y1_1_cut = y1[0][1:x_lim]
    y1_2_cut = y1[1][1:x_lim]
    y1_3_cut = y1[2][1:x_lim]
    y1_4_cut = y1[3][1:x_lim]
    y1_5_cut = y1[4][1:x_lim]
    #print(len(y1_1_cut))
    #print(len(y1_3_cut))
    # mean and stdev
    y1_mean = [statistics.mean(k) for k in zip(y1_1_cut, y1_2_cut, y1_3_cut, y1_4_cut, y1_5_cut)]
    y1_stdev = [statistics.stdev(k) for k in zip(y1_1_cut, y1_2_cut, y1_3_cut, y1_4_cut, y1_5_cut)]
    # dataframe
    df_1 = pd.DataFrame(list(zip(x_plot, y1_mean, y1_stdev)), 
        columns = ['Episode','Mean','Stdev'])
    #display(df_1)
    # again for episode length
    y2_1_cut = y2[0][1:x_lim]
    y2_2_cut = y2[1][1:x_lim]
    y2_3_cut = y2[2][1:x_lim]
    y2_4_cut = y2[3][1:x_lim]
    y2_5_cut = y2[4][1:x_lim]
    # mean and stdev
    y2_mean = [statistics.mean(k) for k in zip(y2_1_cut, y2_2_cut, y2_3_cut, y2_4_cut, y2_5_cut)]
    y2_stdev = [statistics.stdev(k) for k in zip(y2_1_cut, y2_2_cut, y2_3_cut, y2_4_cut, y2_5_cut)]
    # dataframe
    df_2 = pd.DataFrame(list(zip(x_plot, y2_mean, y2_stdev)), 
        columns = ['Episode','Mean','Stdev'])
        #display(df_2)
    # now, we will apply a simple moving average filter, otherwise the graph cannot be read
    moving_window = 5 # 20 for DDPG and 5 for PPO
    df_1['Mean_SMA20'] = df_1['Mean'].rolling(moving_window).mean()
    df_1['Stdev_SMA20'] = df_1['Stdev'].rolling(moving_window).mean()
    # duplicate
    df_2['Mean_SMA20'] = df_2['Mean'].rolling(moving_window).mean()
    df_2['Stdev_SMA20'] = df_2['Stdev'].rolling(moving_window).mean()
    # reduce plot data
    subsample = 20 # subsampling because it is too much data (too dense plot)
    df_1_sub = df_1[::subsample]
    df_2_sub = df_2[::subsample]

    ## time for plots!! -> ativar sns, plt, e cancerizações
    sns.set_theme(style="darkgrid")
    #
    _, ax1 = plt.subplots()
    #
    #plt.rc( 'text', usetex = True ) # TeX dando pau na minha
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams.update({'font.size': 52})

    # plot return
    df_1_sub["Upper_limit"] = df_1_sub["Mean_SMA20"]+df_1_sub["Stdev_SMA20"]
    df_1_sub["Lower_limit"] = df_1_sub["Mean_SMA20"]-df_1_sub["Stdev_SMA20"]
    ln1 = sns.lineplot(x="Episode", y="Mean_SMA20", color='blue', data=df_1_sub, alpha=0.8, label='Average Episodic Return')
    ax1.fill_between(df_1_sub["Episode"], df_1_sub["Lower_limit"] , df_1_sub["Upper_limit"] , color='b', alpha=0.3)
    #ax1.set_title('DDPG Training Results')
    ax1.set_title('PPO Training Results')
    #ax1.set_xlabel('Episode')
    ax1.set_xlabel('Batch Iterations')
    ax1.set_ylabel('Return')
    
    # overplot length
    ax2 = ax1.twinx()
    df_2_sub["Upper_limit"] = df_2_sub["Mean_SMA20"]+df_2_sub["Stdev_SMA20"]
    df_2_sub["Lower_limit"] = df_2_sub["Mean_SMA20"]-df_2_sub["Stdev_SMA20"]
    ln2 = sns.lineplot(x="Episode", y="Mean_SMA20", color='red', data=df_2_sub, alpha=0.8, label='Average Episodic Length')
    ax2.fill_between(df_2_sub["Episode"], df_2_sub["Lower_limit"] , df_2_sub["Upper_limit"] , color='r', alpha=0.3)    
    ax2.set_ylabel('Length')

    # legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.93))

    # save
    plt.grid()
    plt.tight_layout()
    #plt.savefig('./figures/ddpg_training.eps', format='eps') # I saved manually
    plt.show()

    

def main(args):
    ''' Filepaths '''
    
    # ppo
    filepaths = [
        "./training_data/ppo/train_1.txt",
        "./training_data/ppo/train_2.txt",
        "./training_data/ppo/train_3.txt",
        "./training_data/ppo/train_4.txt",
        "./training_data/ppo/train_5.txt",
    ]
    # ddpg
    '''
    filepaths = [
        "./training_data/ddpg/train_1.txt",
        "./training_data/ddpg/train_2.txt",
        "./training_data/ddpg/train_3.txt",
        "./training_data/ddpg/train_4.txt",
        "./training_data/ddpg/train_5.txt",
    ]
    '''
    graph_data(filepaths, args)

if __name__ == '__main__':
    args = get_args()
    main(args)
