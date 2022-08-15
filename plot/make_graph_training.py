import sys
import matplotlib.pyplot as plt

import argparse

def get_args():

    # example: python3 make_graph_training.py --file test_data/test_ppo_actor_1.txt --title 'Test Results' --save

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', type=str, default='')
    parser.add_argument('--title', dest='title', type=str, default='Training Results')
    parser.add_argument('--save', dest='save', action='store_true', help='data to figures/<title>.png')

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

def graph_data(filepath, args):
    ''' Plots Data '''
    
    x, y1, y2 = extract_data(filepath)

    _, ax1 = plt.subplots()

    # first plot
    ln1 = ax1.plot(x, y1, c=[0,0,1], label='Average Episodic Reward')
    ax1.set_title(args.title)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Return')

    # second plot
    ax2 = ax1.twinx()
    ln2 = ax2.plot(x, y2, c=[1,0,0], label='Average Episodic Length')
    ax2.set_ylabel('Time Steps')

    # legend
    ln = ln1+ln2
    lab = [l.get_label() for l in ln]
    ax1.legend(ln, lab, loc=0)

    # schon
    plt.grid(axis='both')
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
