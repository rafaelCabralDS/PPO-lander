import sys
import matplotlib.pyplot as plt

import argparse

import numpy as np

sys.path.append("..") # quick hack -> search in the parent directory
from env.constants import *

'''
Shows one figure at a time.
'''


def get_args():

    # example: python3 flight_profile.py --file test_data/profile_ppo_actor_1.dat --save

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', type=str, default="test_data/profile_ppo_actor_1.dat")
    parser.add_argument('--save', dest='save', action='store_true', help='data to figures/<title>.png')

    args = parser.parse_args()

    return args


# WRITE EXTRACT_DATA AND GRAPH_DATA !!

def extract_data(filepath):
    '''Extract the iteration and episodic return from the logging data '''
    # x is iterations so far, y is average episodic reward
    step, fuel, state, action  = [], [], [], []

    # extract out x's and y's
    with open(filepath, 'r') as f:
        for l in f:
            l = [e.strip() for e in l.split(':')]
            if 'Step' in l:
                step.append(int(l[1]))
            elif 'Fuel' in l:
                fuel.append(float(l[1]))
            elif 'State' in l:
                helper = l[1].split(",")
                state.append([float(element) for element in helper if element != ''])
            elif 'Action' in l:
                helper = l[1].split(",")
                action.append([float(element) for element in helper if element != ''])
    
    return step, fuel, state, action



def graph_data(filepath, args):
    ''' Plots Data '''
    
    step, fuel, state, action = extract_data(filepath)

    # quick print
    #print(f"\nstep: {step}")
    #print(f"\nfuel: {fuel}")
    #print(f"\nstate: {state}")
    #print(f"\naction: {action}")

    # extract pure inputs and outputs
    #fuel is already listed
    dx = [float(element[0]) for element in state]
    dh = [float(element[1]) for element in state]
    vx = [float(element[2]) for element in state]
    vh = [float(element[3]) for element in state]
    theta = [float(element[4]) for element in state]
    q = [float(element[5]) for element in state]
    T_m = [float(element[0]) for element in action]
    T_s = [float(element[1]) for element in action]
    Delta_delta = [float(element[2]) for element in action]
    ## plots
    #plt.rcParams['text.usetex'] = True
    plt.rcParams["legend.loc"] = 'upper right' 

    # fuel with time ---1
    fuel_percentage = [100 * fuel_k / fuel[0] for fuel_k in fuel]
    plt.plot(step, fuel_percentage, 'b')
    plt.title('Remaining Fuel')
    plt.ylabel('Fuel [%]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # controls with time ---2
    # APPLY CLIP FILTERS HERE AND FUEL CHECK
    T_m_clipped, T_s_clipped, Delta_delta_clipped  = [], [], []
    for idk, k in enumerate(T_m):
        if fuel[idk] == 0:
            T_m_clipped.append(0.0)
        else:
            if k > 1.0:
                T_m_clipped.append(1.0)
            elif k < MAIN_ENGINE_LOWER:
                T_m_clipped.append(0.0)
            else:
                T_m_clipped.append(k)
    for idk, k in enumerate(T_s):
        if fuel[idk] == 0:
            T_s_clipped.append(0.0)
        else:
            if k > 1.0:
                T_s_clipped.append(1.0)
            elif k < -1.0:
                T_s_clipped.append(-1.0)
            elif k > -SIDE_ENGINE_ACTIVATE and k < SIDE_ENGINE_ACTIVATE:
                T_s_clipped.append(0.0)
            else:
                T_s_clipped.append(k)
    for k in Delta_delta:
        if k > 1.0:
            Delta_delta_clipped.append(1.0)
        elif k < -1.0:
            Delta_delta_clipped.append(-1.0)
        else:
            Delta_delta_clipped.append(k)
    plt.plot(step, T_m_clipped, 'r')
    plt.plot(step, T_s_clipped, 'b')
    plt.plot(step, Delta_delta_clipped, 'g')
    plt.legend(['T_m','T_s','Delta(delta)$'])
    plt.title('Applied Control Efforts in Time')
    plt.ylabel('Normalized Control Inputs')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # kinematic states -> (x,h) with time ---3
    # CONVERT TO METERS HERE -> invert state expression ignoring W and multiplying by PIXELTOMETER
    dx_m = [dx_k * PIXELTOMETER for dx_k in dx]
    dh_m = [dh_k * PIXELTOMETER for dh_k in dh]
    plt.plot(step, dx_m, 'r')
    plt.plot(step, dh_m, 'b')
    plt.legend(['dx','dh'])
    plt.title('Kinematic States')
    plt.ylabel('Error to Target [m]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # kinematic derivatives -> (\dot{x},\dot{h}) with time ---4
    # CONVERT TO M/S HERE -> invert state expression ignoring W and multiplying by PIXELTOMETER
    vx_m = [vx_k * PIXELTOMETER for vx_k in vx]
    vh_m = [vh_k * PIXELTOMETER for vh_k in vh]
    plt.plot(step, vx_m, 'r')
    plt.plot(step, vh_m, 'b')
    plt.legend(['Vx','Vh'])
    plt.title('Velocity States')
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time step [s/60]')
    plt.ylim(vh_m[0]*1.1,150)
    plt.show()
    plt.draw()

    # theta with time ---5
    # TO DEG
    theta_deg = [theta_k / DEGTORAD for theta_k in theta]
    plt.plot(step, theta_deg, 'b')
    plt.title('Theta Angle')
    plt.ylabel('Theta [Deg]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # \dot{theta} with time ---6
    # TO DEG
    q_deg = [q_k / DEGTORAD for q_k in q]
    plt.plot(step, q_deg, 'b')
    plt.title('Theta Derivative')
    plt.ylabel('q [Deg/s]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # plot (dx,dh) -> trajectory ---7
    plt.plot(dx_m, dh_m, 'b')
    plt.title('Trajectory')
    plt.ylabel('dh [m]')
    plt.xlabel('dx [m]')
    plt.xlim(-10,+10)
    plt.ylim(0,dh_m[0])
    plt.show()
    plt.draw()


def main(args):
    ''' Verifies File and Plots '''
    
    filepath = args.file
    if filepath == '':
        print(f"Didn't specify log file. Use --file <filepath>.", flush=True)
        sys.exit(0)

    graph_data(filepath, args)

if __name__ == '__main__':
    args = get_args()
    main(args)
