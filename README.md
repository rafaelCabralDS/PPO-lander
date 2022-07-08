# rocket-lander-homework
Adaptation of Ferrante's master's environment with distinct rocket dynamics and parameters, plus a PPO rl algorithm to train an actor network as controller, based on Yu's implementation.

* Ferrante's work: https://github.com/arex18/rocket-lander
* Yu's PPO implementation: https://github.com/ericyangyu/PPO-for-Beginners

## Preliminaries

* states: Bidimentional 3-DoF dynamics, with yaw included and ground_contact status on both legs
* PPO controller

## How to start

Start by downloading the required libraries:

```
pip install -r requirements.txt
```

If it does not work, install each dependency separately. Then, run:

```
python3 test.py
```

~~(If we are problemless)~~ **Congratulations**, you have just landed a rocket!! ~~(or atleast tried to)~~

To save data for plotting, just pipe stdout to a file, like:

```
python3 test.py > plot/test_data/test_ppo_actor_1.txt
```

Then, run make_graph.py

## Results

## TODO

* where *************** then pay attention
