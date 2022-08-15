"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network.
"""

import tensorflow as tf

import sys


from train_article_model import train as train_article_model


from exploration import OUPolicy

#sys.path.append('..')

#from ddpg.ddpg import DDPG
#from ddpg.train import set_up

from ddpg import DDPG # a visualização do vscode parece errada, mas é isso mesmo (pq vc está visualizando da raíz)
from train import set_up

sys.path.append('..')
from env.constants import DEGTORAD
from env.rocketlander import RocketLander

FLAGS = set_up()

action_bounds = [1, 1, 15*DEGTORAD]

eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 500}
env = RocketLander(simulation_settings)
#env = wrappers.Monitor(env, '/tmp/contlunarlander', force=True, write_upon_reset=True)

FLAGS.retrain = False # Restore weights if False
FLAGS.test = True
FLAGS.num_episodes = 500
model_dir = './model4' # relative to the current location

# use tensorflow 1 behavior
tf.compat.v1.disable_eager_execution()

with tf.device('/cpu:0'):
    agent = DDPG(
        action_bounds,
        eps,
        env.observation_space.shape[0],
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        retrain=True, # set to true for now  TODO: CHANGE IF NECESSARY
        log_dir='./logs',
        model_dir=model_dir)

    #train(env, agent, FLAGS)
    #test(env, agent, simulation_settings)
    #train_third_model_normalized(env, agent, FLAGS)
    train_article_model(env, agent, FLAGS)
#train_second_model(env, agent, FLAGS)

