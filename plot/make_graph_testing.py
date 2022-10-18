import sys
import matplotlib.pyplot as plt

import argparse

def get_args():

    # example: python3 make_graph_testing.py --file test_data/test_ppo_actor_1.txt --title 'Test Results' --save

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', type=str, default='')
    parser.add_argument('--title', dest='title', type=str, default='DDPG Testing Results') # best: model 3
    #parser.add_argument('--title', dest='title', type=str, default='PPO Testing Results') # best: model 3
    parser.add_argument('--save', dest='save', action='store_true', help='data to figures/<title>.png')

    args = parser.parse_args()

    return args

def extract_data(filepath):
    '''Extract the iteration and episodic return from the logging data '''
    # x is iterations so far, y is average episodic reward, success is for successful landings
    x, y, success = [], [], []


    # goHorse go!
    cut_point = 5000 # plot until here

    # extract out x's and y's
    with open(filepath, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Episodic Return' in l:
                y.append(float(l[1]))
            elif 'Iteration' in l:
                x.append(int(l[1]))
            elif 'Success' in l:
                success.append(int(l[1]))
                if len(success) >= cut_point: # Iteration appears before episodic return
                    return x, y, success
    
    return x, y, success

def graph_data(filepath, args):
    ''' Plots Data '''
    
    x, y, success = extract_data(filepath)

    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.dpi': 200})


    plt.plot(x, y, 'o', c=[0,0,1])
    
    plt.title(args.title)
    plt.xlabel('Episode') # TODO: Iteration Window
    plt.ylabel('Return')
    
    # weirdly has to be before plt.show()
    if args.save:
        plt.savefig(f'figures/{args.title}.png')
    
    plt.grid()
    plt.tight_layout()
    
    # give success results, before plotting data

    success_percentage = sum(success) / len(success) * 100

    print(f"success_percentage = {success_percentage}%")

    plt.show()
    plt.draw()
    

def main(args):
    ''' Verifies File and Plots '''
    
    filepath = args.file
    if filepath == '':
        print(f"Didn't specify log file. Use --file <filepath>.", flush=True)
        sys.exit(0)

    graph_data(filepath, args)

if __name__ == '__main__':
    args = get_args()
    main(args)
