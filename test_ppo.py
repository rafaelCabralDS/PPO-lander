"""
Author: Gabriel S and Octávio M
Date: 05/07/2022
"""
from env.rocketlander import RocketLander
#from env.rocketlander_first_cp import RocketLander

from env.constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT

from PPO_beginner.ppo import PPO
from PPO_beginner.network import FeedForwardNN

def main():
    # Settings holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)
    #env.reset()

#    from control_and_ai.pid import PID_Benchmark
    # Initialize the PID algorithm
#    pid = PID_Benchmark()
    
    hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 2000, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 5
			  }

    # a training section to see if everything running smooth
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    t_to_end = 100000
    model.learn(total_timesteps=t_to_end)

if __name__ == "__main__":
    main()

