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
SCALE = 30

VIEWPORT_W = 1000 
VIEWPORT_H = 800 # ~ 250 m

SEA_CHUNKS = 25

DEGTORAD = math.pi/180.0

W = int(VIEWPORT_W / SCALE)
H = int(VIEWPORT_H / SCALE)

# --------------------------------
'''Rocket Relative Dimensions'''
# changed name to something better
INITIAL_RANDOM_FORCE = 10000.0 # WILL NEED TO CHANGE HERE

RANDOM_DISTURBANCE_FORCE = 2000.0

LANDER_LENGTH = 227 # 70 m
LANDER_RADIUS = 10

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

LEG_AWAY = 30
LEG_DOWN = 0.3
LEG_W, LEG_H = 3, LANDER_LENGTH/8

SIDE_ENGINE_VERTICAL_OFFSET = 5
SIDE_ENGINE_HEIGHT = LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET
SIDE_ENGINE_AWAY = 10.0

CAP_POLY = [
    (-LANDER_RADIUS, 0), (LANDER_RADIUS, 0),
    (1, LANDER_LENGTH/6), (-1, LANDER_LENGTH/6)
]

LEG_LOWEST = 40 * DEGTORAD
LEG_HIGHEST = 45 * DEGTORAD

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

# --------------------------------
'''Forces, Costs, Torque, Friction'''
MAIN_ENGINE_LOWER = 0.3 # 30% of the maximum power # referenced in __main_engines_force_computation
MAIN_ENGINE_POWER = FPS*LANDER_LENGTH / (2.1) # FPS IS A RELEVANTE PARAMETER TO CONTROL SIMULATION

SIDE_ENGINE_ACTIVATE = 0.5 # atleast 50% to enable side thrusters
SIDE_ENGINE_POWER = MAIN_ENGINE_POWER / 50

INITIAL_FUEL_MASS_PERCENTAGE = 0.2
MAIN_ENGINE_FUEL_COST = MAIN_ENGINE_POWER/SIDE_ENGINE_POWER
SIDE_ENGINE_FUEL_COST = 1

LEG_SPRING_TORQUE = LANDER_LENGTH/2
NOZZLE_TORQUE = 500
NOZZLE_ANGLE_LIMIT = 15*DEGTORAD

BARGE_FRICTION = 2 # ??

# --------------------------------
'''Landing Calibration'''
LANDING_VERTICAL_CALIBRATION = 0.03
TERRAIN_CHUNKS = 22 # 0-20 calm seas, 20+ rough seas
BARGE_LENGTH_X1_RATIO = 0.35 # ???
BARGE_LENGTH_X2_RATIO = 0.65 # ???

# --------------------------------
'''Params'''
# *************** VALIDAR, MAS ACHO QUE É REALISTA
MASS = 25.222
L1 = 3.8677
L2 = 3.7
LN = 0.1892
INERTIA = 482.2956
GRAVITY = 9.81

# --------------------------------
'''State Reset Limits'''
THETA_LIMIT = 35*DEGTORAD

# --------------------------------
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


