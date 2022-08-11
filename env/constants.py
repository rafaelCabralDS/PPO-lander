import math
import matplotlib
matplotlib.use('TkAgg')
from enum import Enum


# --------------------------------
'''Simulation Update'''
FPS = 60
UPDATE_TIME = 1/FPS

# --------------------------------
'''Simulation view, scale and unit conversions'''


# width is somewaht free to adjust, height you can't vary too much

VIEWPORT_H = 900
VIEWPORT_W = VIEWPORT_H

LANDER_LENGTH_PIXEL = 50
LANDER_LENGTH_M = 3


# [0,1] -> [0,~~50]
PIXELTOMETER = (VIEWPORT_H/LANDER_LENGTH_PIXEL) * LANDER_LENGTH_M # 54

DEGTORAD = math.pi/180.0

W = int(PIXELTOMETER)
H = int(PIXELTOMETER)


SCALE = PIXELTOMETER

SEA_CHUNKS = 25 # 25
HELIPAD_Y = H / 50
BARGE_HEIGHT = HELIPAD_Y * 1.6

# --------------------------------
'''Rocket Relative Dimensions'''

LANDER_LENGTH = LANDER_LENGTH_M * PIXELTOMETER # -> 154    # 120 # 227 -> A BIT MORE THAN 3 [m] WITH CAP -> 3.5 M WITH LEGS 
LANDER_RADIUS = 7 # 3 # 6 # 10

# changed to look capy
LANDER_POLY = [
    (-LANDER_RADIUS, 0),
    (-LANDER_RADIUS/5, LANDER_LENGTH + LANDER_LENGTH/10), (+LANDER_RADIUS/5, LANDER_LENGTH + LANDER_LENGTH/10), # additional points
    (LANDER_RADIUS, 0),
    (LANDER_RADIUS, LANDER_LENGTH), (-LANDER_RADIUS, LANDER_LENGTH)
]

NOZZLE_POLY = [
    (-LANDER_RADIUS+LANDER_RADIUS/2, 0), (LANDER_RADIUS-LANDER_RADIUS/2, 0), 
    (-LANDER_RADIUS+LANDER_RADIUS/2, LANDER_LENGTH/8), (LANDER_RADIUS-LANDER_RADIUS/2, LANDER_LENGTH/8)
]

LEG_AWAY = 8.5 #6.7  #16 #30
LEG_DOWN = 0.5 #0.12 #0.3

LEG_W, LEG_H = 2.5, LANDER_LENGTH/6

SIDE_ENGINE_VERTICAL_OFFSET = 1.1 # 2.65 # 5
SIDE_ENGINE_HEIGHT = LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET
SIDE_ENGINE_AWAY = LANDER_RADIUS/2+0.01 # 5.3 # 10.0

CAP_POLY = [
    (-LANDER_RADIUS, 0), (LANDER_RADIUS, 0),
    (1, LANDER_LENGTH/6), (-1, LANDER_LENGTH/6)
]

LEG_LOWEST = 30 * DEGTORAD
LEG_HIGHEST = 35 * DEGTORAD

# --------------------------------
'''State array'''
# ***************
class State(Enum):
    x = 0
    y = 1
    x_dot = 2
    y_dot = 3
    theta = 4
    theta_dot = 5
    left_ground_contact = 6
    right_ground_contact = 7
    previous_main_thrust = 8
    previous_side_thrust = 9
    delta = 10

'''State Definition'''
# ***************
# More Readable, to access states
XX = State.x.value
YY = State.y.value
X_DOT = State.x_dot.value
Y_DOT = State.y_dot.value
THETA = State.theta.value
THETA_DOT = State.theta_dot.value
LEFT_GROUND_CONTACT = State.left_ground_contact.value
RIGHT_GROUND_CONTACT = State.right_ground_contact.value
PREVIOUS_MAIN_THRUST = State.previous_main_thrust.value
PREVIOUS_SIDE_THRUST = State.previous_side_thrust.value
DELTA = State.delta.value

# --------------------------------
'''Forces, Costs, Torque, Friction'''
MAIN_ENGINE_LOWER = 0.3 # 0.3 # referenced in __main_engines_force_computation
MAIN_ENGINE_POWER = 1200 #

SIDE_ENGINE_ACTIVATE = 0.5
SIDE_ENGINE_POWER = MAIN_ENGINE_POWER / 50

INITIAL_FUEL_MASS_PERCENTAGE = 0.22
MAIN_ENGINE_FUEL_COST = 10
SIDE_ENGINE_FUEL_COST = 1

LEG_SPRING_TORQUE = LANDER_LENGTH
NOZZLE_TORQUE = 600 # 1000 #
NOZZLE_ANGLE_LIMIT = 20*DEGTORAD # delta + theta

BARGE_FRICTION = 3 #

ANGULAR_VELOCITY_AMPLIFIER = 20.0

# changed name to something better
INITIAL_RANDOM_FORCE = 0.0 # remove this and insert randomness in initial conditions

RANDOM_DISTURBANCE_FORCE = 0.0 # RANDOM
RANDOM_WIND_X = 0.1 # [m/s] ADDITIONAL VARIANCE IN GAUSSIAN IN X
RANDOM_WIND_Y = 0.01  # [m/s] IN Y

# --------------------------------
'''Landing Calibration'''
TERRAIN_CHUNKS = 22 # 0-20 calm seas, 20+ rough seas
BARGE_LENGTH_X1_RATIO = 0.3 #
BARGE_LENGTH_X2_RATIO = 0.7 #

# --------------------------------
'''Params'''
# *************** botar parâmetros que quiser
MASS = 23.32 # [Kg] # = density*area/SCALE^2 #25.222
INERTIA = 73.05 # [Kg^2/m^2] # approx result #482.2956
GRAVITY = 9.80665 #9.81

XCG_NOSE = 1.5

# below are not explicitly required, keeping only for reference (ARE WRONG)
L1 = 3.8677
L2 = 3.7
LN = 0.1892

# --------------------------------
'''State Reset Limits'''
THETA_LIMIT = 35*DEGTORAD



