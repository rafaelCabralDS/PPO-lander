"""
Authors: Gabriel S and Octávio M
Date: 05/07/2022
"""
from env.rocketlander import RocketLander

from program_control.hyperparameter_constants import *

from ppo.ppo import PPO
from ppo.network import FeedForwardNN

def main():
    # Settings holds all the settings for the rocket lander environment.
    settings = {'Side Engines': SIDE_ENGINES,
                'Clouds': CLOUDS,
                'Vectorized Nozzle': VECTORIZED_NOZZLE,
                'Starting Y-Pos Constant': STARTING_Y_POS_CONSTANT,
                'Initial Force': INITIAL_FORCE}  # (6000, -10000)}

    env = RocketLander(settings)

    hyperparameters = {
				'timesteps_per_batch': TIMESTEPS_PER_BATCH, 
				'max_timesteps_per_episode': MAX_TIMESTEPS_PER_EPISODE, 
				'gamma': GAMMA, 
				'n_updates_per_iteration': N_UPDATES_PER_ITERATION,
				'lr': LR, 
				'clip': CLIP,
				'render': RENDER,
				'render_every_i': RENDER_EVERY_I,
                't_to_end' : T_TO_END,
                'save_actor_model' : "'./ppo/models/ppo_actor_0.pth'", # file_name
                'save_critic_model' : "'./ppo/models/ppo_critic_0.pth'"
			  }

    # a training section to see if everything running smooth
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    t_to_end = hyperparameters['t_to_end']
    model.learn(total_timesteps=t_to_end)

if __name__ == "__main__":
    main()

