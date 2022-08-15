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

# width is somewaht free to adjust, height you can't vary too much

VIEWPORT_W = 1200 # 1600 
VIEWPORT_H = 900 # 900 # 1000 # ~ 250 m

PIXELTOMETER = 900/50*3 # Viewport_H [PIXEL] / LANDER_LENGTH [PIXEL] * ROCKET_LENGTH [m] 

DEGTORAD = math.pi/180.0

W = int(VIEWPORT_W / SCALE)
H = int(VIEWPORT_H / SCALE)


SEA_CHUNKS = 25 # 25
HELIPAD_Y = H / 50
BARGE_HEIGHT = HELIPAD_Y * 1.6

# --------------------------------
'''Rocket Relative Dimensions'''
# changed name to something better
INITIAL_RANDOM_FORCE = 0.0 # remove this and insert randomness in initial conditions

RANDOM_DISTURBANCE_FORCE = 0.0 # proportional to altitude

LANDER_LENGTH = 50 # 120 # 227 -> 3 [m]
LANDER_RADIUS = 3 # 6 # 10

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
#    (-LANDER_RADIUS+LANDER_RADIUS/2, 0), (LANDER_RADIUS-LANDER_RADIUS/2, 0), 
#    (-LANDER_RADIUS+LANDER_RADIUS/2, LANDER_LENGTH/8), (LANDER_RADIUS-LANDER_RADIUS/2, LANDER_LENGTH/8)
]

LEG_AWAY = 6.7  #16 #30
LEG_DOWN = 0.12 #0.3

#LEG_W, LEG_H = 3, LANDER_LENGTH/8
LEG_W, LEG_H = 1.0, LANDER_LENGTH/6

SIDE_ENGINE_VERTICAL_OFFSET = 1.1 # 2.65 # 5
SIDE_ENGINE_HEIGHT = LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET
SIDE_ENGINE_AWAY = 2.2 # 5.3 # 10.0

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

# --------------------------------
'''Forces, Costs, Torque, Friction'''
MAIN_ENGINE_LOWER = 0.3 # 30% of the maximum power # referenced in __main_engines_force_computation
MAIN_ENGINE_POWER = FPS*LANDER_LENGTH / (1.5)  #(2.1) # FPS IS A RELEVANTE PARAMETER TO CONTROL SIMULATION

SIDE_ENGINE_ACTIVATE = 0.5 # atleast 50% to enable side thrusters
SIDE_ENGINE_POWER = MAIN_ENGINE_POWER / 50

INITIAL_FUEL_MASS_PERCENTAGE = 0.22
MAIN_ENGINE_FUEL_COST = MAIN_ENGINE_POWER/SIDE_ENGINE_POWER # 50
SIDE_ENGINE_FUEL_COST = 1

LEG_SPRING_TORQUE = LANDER_LENGTH
NOZZLE_TORQUE = 1000 # 250 # 500
NOZZLE_ANGLE_LIMIT = 20*DEGTORAD # THETA + DELTA

BARGE_FRICTION = 2 # ??

ANGULAR_VELOCITY_AMPLIFIER = 20.0

# --------------------------------
'''Landing Calibration'''
TERRAIN_CHUNKS = 22 # 0-20 calm seas, 20+ rough seas
BARGE_LENGTH_X1_RATIO = 0.3 #
BARGE_LENGTH_X2_RATIO = 0.7 #

# --------------------------------
'''Params'''
# *************** botar parâmetros que quiser
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


