'''
    This file contains the arguments to parse at command line in train.py
'''

import argparse

def get_args():

    # example: python3 train.py --load_actor_mode './ppo/models/ppo_actor_1.pth' --load_critic_model './ppo/models/ppo_critic_1.pth' --save_actor_model "'./ppo/models/ppo_actor_1.pth'"  --save_critic_model "'./ppo/models/ppo_critic_1.pth'"

    # to save: > plot/training_data/train_ppo_1.txt

    parser = argparse.ArgumentParser()

    # models
    parser.add_argument('--load_actor_model', dest='load_actor_model', type=str, default='./ppo/models/ppo_actor_0.pth')
    parser.add_argument('--load_critic_model', dest='load_critic_model', type=str, default='./ppo/models/ppo_critic_0.pth')
    parser.add_argument('--save_actor_model', dest='save_actor_model', type=str, default="'./ppo/models/ppo_actor_0.pth'")
    parser.add_argument('--save_critic_model', dest='save_critic_model', type=str, default="'./ppo/models/ppo_critic_0.pth'")
    # other args
    parser.add_argument('--render', dest='render', action='store_true')
    args = parser.parse_args()

    return args


