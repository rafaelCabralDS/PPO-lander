import sys
import os
import shutil
import argparse

import numpy as np

import time

sys.path.append('..')
from env.rocketlander import get_state_sample
from env.constants import *
from utils import Utils



def train(env, agent, FLAGS):

    obs_size = env.observation_space.shape[0]

    util = Utils()
    state_samples = get_state_sample(samples=6000, normal_state=False, untransformed_state=False)
    util.create_normalizer(state_sample=state_samples)

    total_timesteps = 1e7 # same as T_TO_END in hyperparameter_constants

    t_so_far = 0   # total timesteps in training
    episode = 0    # same as i_so_far

    current_t = time.time_ns() # total delta_t
    initial_t = current_t

    while t_so_far < total_timesteps:
        
        episode += 1 # next episode
        
        old_state = None
        done = False
        total_reward = 0

        s = env.reset()
        state = s
                
        max_steps = 900 # same as MAX_TIMESTEPS_PER_EPISODE in hyperparameter_constants

        previous_t_so_far = t_so_far

        for t in range(max_steps): # env.spec.max_episode_steps
            
            t_so_far += 1 # yet another timestep

            if False: # True # direto, depois coloco false
            
                env.render()
                env.draw_marker(x=env.landing_coordinates[0], y=env.landing_coordinates[1], isBody=False) # landing marker
                env.refresh(render=False)

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), not FLAGS.test)

            # take it
            s, reward, done, _ = env.step(action[0])
            state = s
            
            total_reward += reward

            if not FLAGS.test:
                # update q vals
                agent.update(old_state, action[0], np.array(reward), state, done)

            if done:
                break

        #agent.log_data(total_reward, episode)

        if (episode % 50 == 0 or episode == 1) and not FLAGS.test: # salvar a cada 50 parece menos verboso
            #print('Saved model at episode', episode)
            agent.save_model(episode)
        
        # print data
        length = (t_so_far - previous_t_so_far)
        episodic_return = reward
        previous_t = current_t
        current_t = time.time_ns()
        delta_t = (current_t - previous_t) / 1e9
        if not FLAGS.test:
            print_train_output(episode, length, episodic_return, t_so_far, delta_t, initial_t)
        else:
            print_test_output(episode, length, episodic_return, env.successful_landing)
        if episode >= 10001:
            exit()


def print_train_output(iteration, length, episodic_return, t_so_far, delta_t, initial_t):    
    # Print train logging statements
    print(flush=True)
    print(f"----------------------------------------", flush=True)
    print(f"Iteration: {iteration}")
    print(f"Average Episodic Length: {length}", flush=True)
    print(f"Episodic Return: {episodic_return}", flush=True) # AVERAGE
    print(f"Timesteps So Far: {t_so_far}", flush=True)
    print(f"Took: {delta_t} seconds", flush=True)
    print(f"Total Time: {(time.time_ns()-initial_t)/1e9}", flush=True)
    print(f"----------------------------------------", flush=True)

def print_test_output(ep_num, ep_len, ep_ret, successful_landing):
    # Print test logging statements
    print(flush=True)
    print(f"----------------------------------------", flush=True)
    print(f"Iteration: {ep_num}", flush=True) # written as Iteration to simplify make_graph consumption
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"Success: {int(successful_landing)}", flush=True)
    print(f"----------------------------------------", flush=True)