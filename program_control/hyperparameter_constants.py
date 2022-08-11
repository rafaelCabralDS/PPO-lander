'''
    This file contains some training hyperparameters and simulation control definitions
'''

# TRAINING
TIMESTEPS_PER_BATCH = 6000
MAX_TIMESTEPS_PER_EPISODE = 600 # 10 secs of in simulation time if fps = 60
N_UPDATES_PER_ITERATION = 10
# PPO ALGORITHM
GAMMA = 0.99
LR = 3e-4
CLIP = 0.2
# ENVIRONMENT
SIDE_ENGINES = True
CLOUDS = True
VECTORIZED_NOZZLE = True
STARTING_Y_POS_CONSTANT = 1
INITIAL_FORCE = 'random'
# SIMULATION
RENDER = True # only ppo_test.py consumes this
RENDER_EVERY_I = 50
T_TO_END = 1e7



