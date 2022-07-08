# I will rewrite this one 
# Use a parameter to tell how many iterations
# Controle não está funcionando, wtf?

def _log_summary(ep_len, ep_ret, ep_num):
        """ Print to stdout what we've logged so far in the most recent episode """
        # Round decimal places for more aesthetic logging messages
        ep_len = str(round(ep_len, 2))
        ep_ret = str(round(ep_ret, 2))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
        print(f"Episodic Length: {ep_len}", flush=True)
        print(f"Episodic Return: {ep_ret}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

def rollout(policy, env, render):
    """    Returns a generator to roll out each episode given a trained policy and environment """
    # Rollout until user kills process
    while True:
        obs = env.reset()
        done = False

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0            # episodic length
        ep_ret = 0            # episodic return

        while not done:
            t += 1

            if render:
                # rocketlander specific
                env.render()
                env.draw_marker(x=env.landing_coordinates[0], y=env.landing_coordinates[1], isBody=False) # landing marker
                env.refresh(render=False)

            # Query deterministic action from policy and run it
            action = policy(obs).detach().numpy()
            #action = [1,1,1] # teste manual
            print(f"action_vec: {action}")
            obs, rew, done, _ = env.step(action)
            ep_ret += rew
        # end while

        ep_len = t
        # returns episodic length and return in this iteration
        yield ep_len, ep_ret

def eval_policy(policy, env, render=False):
    """ eval_policy will run forever until you kill the process """

    # Rollout with the policy and environment, and log each episode's data
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)