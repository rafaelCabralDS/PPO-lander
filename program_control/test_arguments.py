'''
    This file contains the arguments to parse at command line in test.py
'''

import argparse

def get_args():

    # example: python3 test.py --load_actor_model './ppo/models/ppo_actor_1.pth' --render

    # to save: > plot/test_data/test_ppo_actor_0.txt

    parser = argparse.ArgumentParser()

    # actor model
    parser.add_argument('--load_actor_model', dest='load_actor_model', type=str, default='./ppo/models/ppo_actor_1.pth')
    # animation
    parser.add_argument('--render', dest='render', action='store_true')
    ## plots
    # a single run
    parser.add_argument('--profile', dest='profile', action='store_true')
    # multiple runs -> zero means this is not a monte_carlo run
    parser.add_argument('--monte_carlo', dest='monte_carlo', type=int, default=0, help='number of iterations')

    args = parser.parse_args()

    return args


