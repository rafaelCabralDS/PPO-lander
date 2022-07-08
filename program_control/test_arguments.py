'''
    This file contains the arguments to parse at command line in test.py
'''
## TODO check what else might be nice to add here

import argparse

from traitlets import Bool

def get_args():

    parser = argparse.ArgumentParser()

    # models
    parser.add_argument('--load_actor_model', dest='load_actor_model', type=str, default='./ppo/models/ppo_actor_0.pth')
    #parser.add_argument('--load_critic_model', dest='load_critic_model', type=str, default='ppo/models/ppo_critic_0.pth')
    # other args
    parser.add_argument('--render', dest='render', type=Bool, default=False)

    args = parser.parse_args()

    return args


