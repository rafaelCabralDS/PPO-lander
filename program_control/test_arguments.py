'''
    This file contains the arguments to parse at command line in test.py
'''

import argparse

def get_args():

    # example: python3 test.py --load_actor_model './ppo/models/ppo_actor_1.pth' --render

    # to save: > plot/test_data/test_ppo_actor_0.txt

    parser = argparse.ArgumentParser()

    # models
    parser.add_argument('--load_actor_model', dest='load_actor_model', type=str, default='./ppo/models/ppo_actor_1.pth')
    #parser.add_argument('--load_critic_model', dest='load_critic_model', type=str, default='ppo/models/ppo_critic_0.pth')
    # other args
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()

    return args


