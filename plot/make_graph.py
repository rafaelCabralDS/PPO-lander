import sys
import matplotlib.pyplot as plt

import argparse

def get_args():

    # example: python3 make_graph.py --file test_data/test_ppo_actor_1.txt --title 'Test Results' --save

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', type=str, default='')
    parser.add_argument('--title', dest='title', type=str, default='Results')
    parser.add_argument('--save', dest='save', action='store_true', help='data to figures/<title>.png')

    args = parser.parse_args()

    return args

def extract_data(filepath):
    '''Extract the iteration and episodic return from the logging data '''
    # x is iterations so far, y is average episodic reward
    x, y = [], []

    # extract out x's and y's
    with open(filepath, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Episodic Return' in l:
                y.append(float(l[1]))
            if 'Iteration' in l:
                x.append(int(l[1]))
    
    return x, y

def graph_data(filepath, args):
    ''' Plots Data '''
    
    x, y = extract_data(filepath)

    plt.plot(x, y, 'b')
    
    plt.title(args.title)
    plt.xlabel('Iteration')
    plt.ylabel('Return')
    
    # weirdly has to be before plt.show()
    if args.save:
        plt.savefig(f'figures/{args.title}.png')
    
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
