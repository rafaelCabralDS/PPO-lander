import sys
import matplotlib.pyplot as plt

import argparse

from numpy import random

from statistics import mean, stdev

sys.path.append("..") # quick hack -> search in the parent directory
from env.constants import *

'''
Shows one figure at a time.
'''

def get_args():

    # example: python3 flight_profile.py --file test_data/profile_ppo_actor_1.dat --save

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', type=str, default="test_data/monte_carlo_ppo_actor_1.dat")
    parser.add_argument('--save', dest='save', action='store_true', help='data to figures/<title>.png')

    args = parser.parse_args()

    return args


# WRITE EXTRACT_DATA AND GRAPH_DATA !!

def extract_data(filepath):
    '''Extract the iteration and episodic return from the logging data '''
    # x is iterations so far, y is average episodic reward
    step, fuel, state, action  = [], [], [], [] # iteration properties
    iteration, reward, success = [], [], []     # monte carlo properties
    # extract out x's and y's
    with open(filepath, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Step' in l:
                step.append(int(l[1]))
            elif 'Fuel' in l:
                fuel.append(float(l[1]))
            elif 'State' in l:
                helper = l[1].split(",")
                state.append([float(element) for element in helper if element != ''])
            elif 'Action' in l:
                helper = l[1].split(",")
                action.append([float(element) for element in helper if element != ''])
            # monte carlo props
            elif 'Iteration' in l:
                iteration.append(int(l[1]))
            elif 'Episodic Return' in l:
                reward.append(float(l[1]))
            elif 'Success' in l:
                success.append(int(l[1]))            
    
    return step, fuel, state, action, iteration, reward, success



def graph_data(filepath, args):
    ''' Plots Data '''
    
    step, fuel, state, _, iteration, reward, success = extract_data(filepath)

    # extracting and filtering stuff
    dx = [float(element[0]) for element in state]
    dh = [float(element[1]) for element in state]
    dx_m = [dx_k * PIXELTOMETER for dx_k in dx]
    dh_m = [dh_k * PIXELTOMETER for dh_k in dh]

    # final iteration fuel
    fuel_end = []

    # graphing logic
    counter = 0
    start = 0
    for _ in iteration[0:-1]:
        previous_step = 0
        step_k = step[counter]
        ite_color = random.rand(3,)
        while previous_step <= step_k: # means an iteration is done
        
            # step
            counter += 1
            previous_step = step_k
            step_k = step[counter]
        # end while
        end = counter
        # plot
        plt.plot(dx_m[start:end],dh_m[start:end],c=ite_color)
        start = end

        # extract final fuel
        fuel_end.append(fuel[counter-1])

    # end for
    
    plt.title(f'Monte Carlo Trajectory for {iteration[-1]} iterations')
    plt.ylabel('dh [m]')
    plt.xlabel('dx [m]')
    plt.xlim(-10,+10)
    plt.ylim(0,55)
    plt.show()

    ## stats -> fuel, success, reward
    fuel_percentage = [100 * fuel_k / fuel[0] for fuel_k in fuel_end]
    fuel_mean = mean(fuel_percentage)
    fuel_std = stdev(fuel_percentage)
    reward_mean = mean(reward)
    reward_std = stdev(fuel)
    success_percentage = sum(success) / len(success) * 100
    print(f'''
    fuel_mean = {fuel_mean}
    fuel_std = {fuel_std}
    reward_mean = {reward_mean}
    reward_std = {reward_std}
    success_percentage = {success_percentage}
    ''')

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
