import sys
import torch

from program_control.test_arguments import get_args
from program_control.hyperparameter_constants import *

from env.rocketlander import RocketLander

from ppo.network import FeedForwardNN
from ppo.eval_policy import eval_policy

def test(env, actor_model, render):
    ''' Tests the model '''
    print(f"Testing {actor_model}", flush=True)

    if actor_model == '':
        print(f"Didn't specify model file. Use --load_actor_model <filepath>.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we built our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=env, render=render)

def main(args):
    settings = {'Side Engines': SIDE_ENGINES,
                'Clouds': CLOUDS,
                'Vectorized Nozzle': VECTORIZED_NOZZLE,
                'Starting Y-Pos Constant': STARTING_Y_POS_CONSTANT,
                'Initial Force': INITIAL_FORCE}

    env = RocketLander(settings)

    test(env=env,actor_model=args.load_actor_model, render=args.render)

if __name__ == '__main__':
    args = get_args()
    print(args) # will go to stdout and be saved in the log file -if redirected-
    main(args)
