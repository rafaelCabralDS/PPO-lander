'''
    This file contains the arguments to parse at command line in train.py
'''
## TODO check what else might be nice to add here

import argparse

from traitlets import Bool

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')
    parser.add_argument('--make_graph', dest='make_graph', type=Bool, default=False)
    parser.add_argument('--render', dest='render', type=Bool, default=False)

    args = parser.parse_args()

    return args


