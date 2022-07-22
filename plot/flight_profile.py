'''
TODO: EXTRACT FILTERED ACTIONS (STATE[8],STATE[9]) AND PLOT STATE[10]
'''


import sys
import matplotlib.pyplot as plt

import argparse

sys.path.append("..") # quick hack -> search in the parent directory
from env.constants import *

'''
Shows one figure at a time.
'''


def get_args():

    # example: python3 flight_profile.py --file test_data/profile_ppo_actor_1.dat

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', dest='file', type=str, default="test_data/profile_0.dat")

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
    
    T_m_filtered = [float(element[8]) for element in state]
    T_s_filtered = [float(element[9]) for element in state]
    delta_Total = [float(element[10]) for element in state]

    T_m_output = [float(element[0]) for element in action]
    T_s_output = [float(element[1]) for element in action]
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
    plt.plot(step, T_m_filtered, 'r')
    plt.plot(step, T_s_filtered, 'b')
    plt.legend(['T_m','T_s'])  # r'$\Delta(\delta)$']
    plt.title('Applied Control Efforts in Time')
    plt.ylabel('Normalized Control Inputs')
    plt.xlabel('Time step [s/60]')
    plt.ylim(-1,1)
    plt.show()
    plt.draw()

    # commanded controls with time ---3
    Delta_delta_clipped = []
    for k in Delta_delta:
        if k > 1.0:
            Delta_delta_clipped.append(1.0)
        elif k < -1.0:
            Delta_delta_clipped.append(-1.0)
        else:
            Delta_delta_clipped.append(k)
    plt.plot(step, T_m_output, 'r')
    plt.plot(step, T_s_output, 'b')
    plt.plot(step, Delta_delta_clipped, 'g')
    plt.legend(['T_m','T_s',r'$\Delta(\delta)$'])
    plt.title('Neural Network Output Efforts in Time')
    plt.ylabel('Prefiltered Normalized Control Inputs')
    plt.xlabel('Time step [s/60]')
    plt.ylim(-1,1)
    plt.show()
    plt.draw()

    # delta total with time --4
    delta_Total_deg = [delta_Total_k / DEGTORAD for delta_Total_k in delta_Total]
    plt.plot(step, delta_Total_deg, 'g')
    plt.title(r'Total thrust deflection $\delta$')
    plt.ylabel(r'Angle $\delta$ [°]')
    plt.xlabel('Time step [s/60]')
    plt.ylim(-15,+15)
    plt.show()
    plt.draw()

    # kinematic states -> (x,h) with time ---5
    # CONVERT TO METERS HERE -> invert state expression ignoring W and multiplying by PIXELTOMETER
    dx_m = [dx_k * PIXELTOMETER for dx_k in dx]
    dh_m = [dh_k * PIXELTOMETER for dh_k in dh]
    plt.plot(step, dx_m, 'r')
    plt.plot(step, dh_m, 'b')
    plt.legend([r'$\Delta$x',r'$\Delta$h'])
    plt.title('Kinematic States')
    plt.ylabel('Error to Target [m]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # kinematic derivatives -> (\dot{x},\dot{h}) with time ---6
    # CONVERT TO M/S HERE -> invert state expression ignoring W and multiplying by PIXELTOMETER
    vx_m = [vx_k * FPS / H for vx_k in vx] # * PIXELTOMETER 
    vh_m = [vh_k * FPS / H for vh_k in vh]
    plt.plot(step, vx_m, 'r')
    plt.plot(step, vh_m, 'b')
    plt.legend(['Vx','Vh'])
    plt.title('Velocity States')
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time step [s/60]')
    plt.ylim(-20,5)
    plt.show()
    plt.draw()

    # theta with time ---7
    # TO DEG
    theta_deg = [theta_k / DEGTORAD for theta_k in theta]
    plt.plot(step, theta_deg, 'b')
    plt.title(r'$\theta$ Angle')
    plt.ylabel(r'$\theta$ [°]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # \dot{theta} with time ---8
    # TO DEG
    q_deg = [q_k / DEGTORAD for q_k in q]
    plt.plot(step, q_deg, 'b')
    plt.title(r'$\theta$ Derivative')
    plt.ylabel('q [°/s]')
    plt.xlabel('Time step [s/60]')
    plt.show()
    plt.draw()

    # plot (dx,dh) -> trajectory ---9
    plt.plot(dx_m, dh_m, 'b')
    plt.title('Trajectory')
    plt.ylabel(r'$\Delta$h [m]')
    plt.xlabel(r'$\Delta$x [m]')
    plt.xlim(-10,+10)
    plt.ylim(0,dh_m[0]+2)
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
