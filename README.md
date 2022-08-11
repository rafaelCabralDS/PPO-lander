# rocket-lander-homework
A rocket landing simulation and control project, slightly based on Ferrante's master's environment, plus a PPO RL algorithm to train an actor network as controller, based on Yu's implementation.

* Ferrante's work: https://github.com/arex18/rocket-lander
* Yu's PPO implementation: https://github.com/ericyangyu/PPO-for-Beginners

## Notes

* observations: Bidimentional 3-DoF dynamics, with yaw included and ground_contact status on both legs
* actions: Main Thrust, Side Thrust and Nozzle Angle.
* A PPO agent learns a policy and value function by iterative training.
* The obtained controller was assessed (works!).

## How to start

To work without hacking the code, you will need a screen which can contain a 900x900 window. Else go into env/constants.py and change VIEWPORT_W and VIEWPORT_H to match your screen.

Start by downloading the required libraries:

```
pip3 install -r requirements.txt
```

If it does not work, install each dependency separately. Then, run from the parent folder:

```
python3 test.py --render
```

~~(If we are problemless)~~ **Congratulations**, you have just landed a rocket!! ~~(or atleast tried to)~~

## Results

Go play around in the plot folder, if you want more details than provided in the paper.

## TODO

* We will finish implementing dynamic details.
* We will train an agent until convergence.
* We will correct the necessary points in the article and publish.

## BIZU

Was used to convert from tf_1 to tf_2 compatible. Then manually adjusted for some functions.

'''
$tf_upgrade_v2 \
--intree my_project/ \
--outtree my_project_v2/ \
--reportfile report.txt
'''

## HOW TO TRAIN FOR ARTICLE

Training 5 DDPG models in parallel (1 CPU core each). Run for ddpg folder.

'''
python3 train_ddpg1.py > '../plot/training_data/ddpg/train_1.txt'
python3 train_ddpg2.py > '../plot/training_data/ddpg/train_2.txt'
python3 train_ddpg3.py > '../plot/training_data/ddpg/train_3.txt'
python3 train_ddpg4.py > '../plot/training_data/ddpg/train_4.txt'
python3 train_ddpg5.py > '../plot/training_data/ddpg/train_5.txt'
'''

Training 5 PPO models in parallel (1 CPU core earch). Run from project root folder.

'''
python3 train.py --save_actor_model "'./ppo/models/ppo_actor_1.pth'" --save_critic_model "'./ppo/models/ppo_critic_1.pth'" > './plot/training_data/ppo/train_1.txt'
python3 train.py --save_actor_model "'./ppo/models/ppo_actor_2.pth'" --save_critic_model "'./ppo/models/ppo_critic_2.pth'" > './plot/training_data/ppo/train_2.txt'
python3 train.py --save_actor_model "'./ppo/models/ppo_actor_3.pth'" --save_critic_model "'./ppo/models/ppo_critic_3.pth'" > './plot/training_data/ppo/train_3.txt'
python3 train.py --save_actor_model "'./ppo/models/ppo_actor_4.pth'" --save_critic_model "'./ppo/models/ppo_critic_4.pth'" > './plot/training_data/ppo/train_4.txt'
python3 train.py --save_actor_model "'./ppo/models/ppo_actor_5.pth'" --save_critic_model "'./ppo/models/ppo_critic_5.pth'" > './plot/training_data/ppo/train_5.txt'

'''

BE VERY CAREFUL AS TO NOT OVERWRITE/DELETE THE MODELS OR LOGS. PUSH AND MAKE LOCAL COPIES
