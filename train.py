import sys
import torch

from env.rocketlander import RocketLander

from program_control.hyperparameter_constants import *
from program_control.train_arguments import get_args

from ppo.ppo import PPO
from ppo.network import FeedForwardNN

def train(env, hyperparameters, load_actor_model, load_critic_model):
    print(f"Training Selected", flush=True)

    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    #print(f'observation shape: {model.env.observation_space.shape}')
    #print(f'action shape: {model.env.action_space.shape}')

    # load only exisiting files
    if load_actor_model != '' and load_critic_model != '':
        print(f"Loading in {load_actor_model} and {load_critic_model}...\n", flush=True)
        model.actor.load_state_dict(torch.load(load_actor_model), strict=False)
        model.critic.load_state_dict(torch.load(load_critic_model), strict=False)
        print(f"Successfully loaded.", flush=True)
    elif load_actor_model != '' or load_critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn(total_timesteps=hyperparameters['t_to_end'])

def main(args):
    settings = {'Side Engines': SIDE_ENGINES,
                'Clouds': CLOUDS,
                'Vectorized Nozzle': VECTORIZED_NOZZLE,
                'Starting Y-Pos Constant': STARTING_Y_POS_CONSTANT,
                'Initial Force': INITIAL_FORCE}

    env = RocketLander(settings)

    hyperparameters = {
                'timesteps_per_batch': TIMESTEPS_PER_BATCH, 
                'max_timesteps_per_episode': MAX_TIMESTEPS_PER_EPISODE, 
                'gamma': GAMMA, 
                'n_updates_per_iteration': N_UPDATES_PER_ITERATION,
                'lr': LR, 
                'clip': CLIP,
                'render': args.render,
                'render_every_i': RENDER_EVERY_I,
                't_to_end' : T_TO_END,
                'save_actor_model' : args.save_actor_model, # file_name
                'save_critic_model' : args.save_critic_model
              }
    
    train(env=env, hyperparameters=hyperparameters,
        load_actor_model=args.load_actor_model, load_critic_model=args.load_critic_model)

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)