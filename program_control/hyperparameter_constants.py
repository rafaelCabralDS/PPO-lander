'''
    This file contains some training hyperparameters and simulation control definitions
'''

# TRAINING
TIMESTEPS_PER_BATCH = 2048
MAX_TIMESTEPS_PER_EPISODE = 200
N_UPDATES_PER_ITERATION = 10
# PPO ALGORITHM
GAMMA = 0.99
LR = 3e-4
CLIP = 0.2
# SIMULATION
RENDER_EVERY_I = 50



