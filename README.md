# rocket-lander-homework
A rocket landing simulation and control project, slightly based on Ferrante's master's environment and DDPG algorithm, plus a PPO RL algorithm to train an actor network as controller, based on Yu's implementation.

* Ferrante's work: https://github.com/arex18/rocket-lander
* Yu's PPO implementation: https://github.com/ericyangyu/PPO-for-Beginners

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

