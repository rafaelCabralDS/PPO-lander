'''
TODO: 
- CHANGE FUNCTIONS:
    1 - _main_engines_force_computation
    2 - _side_engines_force_computation
    3 - _aerodynamic_force_computation
    4 - _create_rocket
    5 - _decrease_mass
'''

import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy import interpolate

import random

import os

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import logging
import pyglet
from itertools import chain

from .constants import *


# This contact detector is equivalent the one implemented in Lunar Lander
class ContactDetector(contactListener):
    '''
    Creates a contact listener to check when the rocket touches down.
    '''
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                #print("contacted ground")
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class RocketLander(gym.Env):
    '''
    Continuous Landing of a rocket.
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    # ----------------------------------------------------------------------------
    def __init__(self, settings):
        self._seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, -GRAVITY))
        self.main_base = None
        self.barge_base = None
        self.CONTACT_FLAG = False
        self.count_ticks_to_end = 0

        self.minimum_barge_height = 0
        self.maximum_barge_height = 0
        self.landing_coordinates = []

        self.lander = None
        self.particles = []
        self.state = []
        self.prev_shaping = None

        if settings.get('Observation Space Size'):
            self.observation_space = spaces.Box(-np.inf, +np.inf, (settings.get('Observation Space Size'),))
        else:
            self.observation_space = spaces.Box(-np.inf, +np.inf, (11,)) # 6 kinematic + 2 contact + 3 action filtered
        self.lander_tilt_angle_limit = THETA_LIMIT

        self.game_over = False
    
        self.settings = settings
        self.dynamicLabels = {}
        self.staticLabels = {}

        self.action_space = spaces.Box(-np.inf, +np.inf, (3,)) # Main Engine [0,1], Left/Right Engine [-1,1], Nozzle Angle [-1,1] -> Gradient -> \Delta(\delta)

        self.untransformed_state = [0] * 6  # Non-normalized state

        # aerodynamic table of params, capture, and interpolation
        dirname = os.path.dirname(__file__)
        self.table_AED = pd.read_csv(dirname+'/tables/AED_coeff_datcom_ct_213_lander.csv')
        self.table_atmosphere = pd.read_csv(dirname+'/tables/atmosphere_properties.csv')
        self.table_wind_model = pd.read_csv(dirname+'/tables/wind_models.csv')
        self._get_aed_tables()
        #

        self.reset()

    ''' INHERITED '''

    # ----------------------------------------------------------------------------
    def _seed(self, seed=None):
        self.np_random, returned_seed = seeding.np_random(seed)
        return returned_seed

    # ----------------------------------------------------------------------------
    def reset(self):
        self.destroy()
        self.game_over = False
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.count_ticks_to_end = 0

        smoothed_terrain_edges, terrain_divider_coordinates_x = self._create_terrain(TERRAIN_CHUNKS)

        self.initial_mass = 0
        self.remaining_fuel = 0
        self.prev_shaping = 0
        self.CONTACT_FLAG = False

        self.successful_landing = False

        # Wind
        self.wind_direction = random.choice([True, False]) # left or rigt wind
        self.wind_counter = 0    # aerodynamics randomness
        self.wind_sample = FPS/4 # every quarter second
        self.wind_disturbance = (0, 0) # process noise (wind variation)

        # Engine Stats
        self.action_history = []
        self.action_history.append([0.0, 0.0, 0.0])

        # Reference y-trajectory
        self.y_pos_ref = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
        self.y_pos_speed = [-1.9, -1.8, -1.64, -1.5, -1.5, -1.3, -1.0, -0.9]
        self.y_pos_flags = [False for _ in self.y_pos_ref]

        # Create the simulation objects
        self._create_clouds()
        self._create_barge()
        self._create_base_static_edges(TERRAIN_CHUNKS, smoothed_terrain_edges, terrain_divider_coordinates_x)

        # Adjust the initial coordinates of the rocket
        x = 0.5 + np.random.uniform(-0.13,0.13) # not dx [0,1]
        y = 0.985 # not dy [0,1]
        x_dot = np.random.uniform(-4,+4) # [m/s]
        y_dot = - 9 + np.random.uniform(-2,+2) # [m/s]
        theta = np.random.uniform(-6*DEGTORAD,6*DEGTORAD) # [rad]
        theta_dot = np.random.uniform(-2.5*DEGTORAD,2.5*DEGTORAD) # [rad]

        self._create_rocket((x*W, y*H))
        self.adjust_dynamics(y_dot=y_dot, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

        '''
        # quickly check params for mass and inertia (keep your sanity)

        ##print(f"self.lander.fixtures: {self.lander.fixtures}") # big stats
        print(f"self.lander.mass: {self.lander.mass}")
        print(f"self.lander.inertia: {self.lander.inertia}")
        print(f"self.legs[0].mass: {self.legs[0].mass}\t self.legs[1].mass: {self.legs[1].mass}")
        print(f"self.legs[0].inertia: {self.legs[0].inertia}\t self.legs[1].inertia: {self.legs[1].inertia}")
        print(f"self.nozzle.mass: {self.nozzle.mass}")
        print(f"self.nozzle.inertia: {self.nozzle.inertia}")
        print(f"Total mass: {self.lander.mass+self.legs[0].mass+self.legs[1].mass+self.nozzle.mass}")
        print(f"(Approx) Rocket pitch inertia relativo to CG: {self.lander.inertia+self.legs[0].inertia+self.legs[1].inertia+self.nozzle.inertia+(self.legs[0].mass+self.legs[1].mass)*(1.55)**2 + self.nozzle.mass*(1.5)**2}") # teorema de steiner
        exit(0)
        '''

        # Step through one action = [0, 0, 0] and return the state, reward etc.
        return self.step(np.array([0, 0, 0]))[0]

    # ----------------------------------------------------------------------------
    def destroy(self):
        if not self.main_base: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.main_base)
        self.main_base = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.count_ticks_to_end = 0

    # ----------------------------------------------------------------------------
    def step(self, action): ####

        assert(len(action) == 3)  # Fe, Fs, Delta psi

        reward = 0

        # Check for contact with the ground
        if (self.legs[0].ground_contact or self.legs[1].ground_contact) and self.CONTACT_FLAG == False:
            self.CONTACT_FLAG = True # will end shortly after touching
            if abs(self.state[XX]) <= 0.04: # bonus for precision
                reward += 10
            if abs(self.state[Y_DOT]) >= 8: # punishment for contact speed
                reward += -15

        # Shutdown all Engines upon contact with the ground
        if self.CONTACT_FLAG:
            action = [0, 0, 0]

        if self.settings.get('Vectorized Nozzle'):
            part = self.nozzle
            part.angle = self.lander.angle + float(action[2]) # 
            if part.angle > NOZZLE_ANGLE_LIMIT:
                part.angle = NOZZLE_ANGLE_LIMIT
            elif part.angle < -NOZZLE_ANGLE_LIMIT:
                part.angle = -NOZZLE_ANGLE_LIMIT
        else:
            part = self.lander

        # if fuel is over
        done = False
        if self.remaining_fuel == 0:
            action = [0, 0, action[2]]
            if self.state[YY] > 0.05: # approx 2 m
                done = True
                #logging.info("Oh nein, fuel ist over. Und we are hoch.")
                reward += -25

        # Main Force Calculations -> Thrust and Aero
        m_power = self._main_engines_force_computation(action, rocketPart=part)
        s_power, engine_dir = self._side_engines_force_computation(action)
        ### aerodynamics
        self._aerodynamic_force_computation()

        # Gather Stats and feedback last control                  # total delta
        self.action_history.append([m_power, s_power * engine_dir, part.angle - self.lander.angle])

        # Spending mass to get propulsion
        self._decrease_mass(m_power, s_power)

        # State Vector
        self.previous_state = self.state  # Keep a record of the previous state
        self.state, self.untransformed_state = self._generate_state()  # Generate state
        
        # Rewards for reinforcement learning
        reward += self._compute_rewards(self.state, self.previous_state)  # part angle can be used as part of the reward

        # Check if the game is done, adjust reward based on the final state of the body
        state_reset_conditions = [
            self.game_over,  # Evaluated depending on body contact
            abs(self.state[XX]) >= 0.7,  # Rocket moves out of x-space
            self.state[YY] < 0 or self.state[YY] > 1.0,  # Rocket moves out of y-space or below barge
            abs(self.state[THETA]) > THETA_LIMIT]  # Rocket tilts greater than the "controllable" limit
        if any(state_reset_conditions):
            #print(f"done for state_reset_conditions.")
            done = True
            reward += -50
            #logging.info("Not Nice.")
        if self.CONTACT_FLAG:
            self.count_ticks_to_end += 1
            if self.count_ticks_to_end >= FPS: # 1 second
                done = True
                reward += 15
                self.successful_landing = True

        self._update_particles()

        return np.array(self.state), reward, done, {}  # {} = info (required by parent class)

    ''' PROBLEM SPECIFIC -> PHYSICS, STATES, REWARDS'''

    # ----------------------------------------------------------------------------
    def _main_engines_force_computation(self, action, rocketPart, *args):
        
        # Nozzle Angle Adjustment

        # For readability
        sin = math.sin(rocketPart.angle)
        cos = math.cos(rocketPart.angle)

        # Main engine
        m_power = 0
        try:
            if (action[0] >= MAIN_ENGINE_LOWER):
                # Limits
                #m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.3  # WAS WRONG
                m_power = np.clip(action[0], MAIN_ENGINE_LOWER, 1.0) # NOW CLIPPING CORRECTLY
                assert m_power >= MAIN_ENGINE_LOWER and m_power <= 1.0
                
                # ******************************* TO CORRECT
                ox = sin
                oy = -cos
                impulse_pos = (rocketPart.position[0], rocketPart.position[1] - 0.5)

                # rocketParticles are just a decoration, 3.5 is here to make rocketParticle speed adequate
                p = self._create_particle(10, impulse_pos[0], impulse_pos[1], m_power,
                                          radius=3.5)

                rocketParticleImpulse = (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power)
                bodyImpulse = (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power)
                point = impulse_pos

                # Force instead of impulse. This enables proper scaling and values in Newtons
                p.ApplyForce(rocketParticleImpulse, point, True)
                rocketPart.ApplyForce(bodyImpulse, point, True)
        except:
            print("Error in main engine power.")

        return m_power

    # ----------------------------------------------------------------------------
    def _side_engines_force_computation(self, action):
        
        # Side engines
        sin = math.sin(self.lander.angle)  # for readability
        cos = math.cos(self.lander.angle)

        # side engine
        s_power = 0.0
        y_dir = 1 # Positioning for the side Thrusters
        engine_dir = 0
    
        if (self.settings['Side Engines'] and np.abs(action[1]) > SIDE_ENGINE_ACTIVATE): # Have to be > 0.5
                # Orientation engines
                engine_dir = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), SIDE_ENGINE_ACTIVATE, 1.0)
                assert s_power >= SIDE_ENGINE_ACTIVATE and s_power <= 1.0

                # Positioning
                constant = (LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET) / SCALE
                dx_part1 = - sin * constant  # Used as reference for dy
                dx_part2 = - cos * engine_dir * SIDE_ENGINE_AWAY / SCALE
                dx = dx_part1 + dx_part2

                dy = np.sqrt(
                    np.square(constant) - np.square(dx_part1)) * y_dir - sin * engine_dir * SIDE_ENGINE_AWAY / SCALE

                # Force magnitude
                oy = sin
                ox = cos * (engine_dir)

                # Impulse Position
                impulse_pos = (self.lander.position[0] + dx,
                               self.lander.position[1] + dy)

                try:
                    p = self._create_particle(12, impulse_pos[0], impulse_pos[1], s_power, radius=1.5)
                    p.ApplyForce((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
                    self.lander.ApplyForce((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)
                except:
                    logging.error("Error due to Nan in calculating y during sqrt(l^2 - x^2). "
                                  "x^2 > l^2 due to approximations on the order of approximately 1e-15.")

        return s_power, engine_dir

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    def _aerodynamic_force_computation(self):

        sin_theta = math.sin(self.lander.angle)
        cos_theta = math.cos(self.lander.angle)
        # height
        h_agl = self.state[YY] * PIXELTOMETER
        # ground speed
        vx = self.state[X_DOT] * FPS / H
        vh = self.state[Y_DOT] * FPS / H
        v_in = np.array([vx, vh])
        # wind
        wind = self.wind_direction*self.__compute_wind(h_agl)
        # Wind Process Noise Randomness (Gusts and such disturbances) -> will remain the same value until next if true
        self.wind_counter += 1
        if self.wind_counter % self.wind_sample == 0:
            self.wind_disturbance = (np.random.normal(0, (wind/4)**2 + RANDOM_WIND_X), 
                                np.random.normal(0, RANDOM_WIND_Y)) # [N] conserves until next sample
            self.wind_counter = 0
        wind_in = np.array([wind + self.wind_disturbance[0], 0 + self.wind_disturbance[1]])
        # aerodynamic speed
        v_tas = v_in - wind_in
        # get atmosphere properties
        a_sound, rho = self.__get_atm_properties(h_agl)
        # rocket dimensions for efforts
        L_ref = LANDER_RADIUS / PIXELTOMETER
        S_ref = np.pi * (L_ref)**2
        # mach
        norm_v_tas = LA.norm(v_tas)
        mach = v_tas/a_sound
        # to get v_tas in body frame
        v_tas_b_x = vx * sin_theta + vh * cos_theta
        v_tas_b_z = vx * cos_theta - vh * sin_theta
        # get aoa
        alpha_deg = (np.arctan(v_tas_b_z/v_tas_b_x)) # [rad]
        # compute aerodynamic forces using alpha_deg
        ca, cn, xcp = self.__compute_aed_coeff(mach, alpha_deg)
        # CP to CG
        center_of_pressure_to_cg = xcp - XCG_NOSE # is this right?
        aed_impulse_pos = (self.lander.position[0] + center_of_pressure_to_cg * sin_theta,
                        self.lander.position[1] + center_of_pressure_to_cg * cos_theta)
        # forces
        axial_force_datcom = 0.5 * rho * S_ref * norm_v_tas**2 * ca
        normal_force_datcom = 0.5 * rho * S_ref * norm_v_tas**2 * cn
        # body frame forces
        f_bx = -axial_force_datcom
        f_bz = -normal_force_datcom
        # aerodynamic forces in inertial frame
        f_aed_x = sin_theta*f_bx + cos_theta*f_bz# ----------------------------------------------------------------------------
    
        f_aed_h = cos_theta*f_bx - sin_theta*f_bz
        aed_force = (f_aed_x, f_aed_h)
        # Apply
        self.lander.ApplyForce(aed_impulse_pos, aed_force, True) # ([m],[m]), ([N],[N])

        return
    # ----------------------------------------------------------------------------
    def __compute_wind(self, h_agl):
        
        wind_curr = np.interp(h_agl, self.h_agl_wind_vec, self.wind_shear_vec)
        
        return wind_curr
    # ----------------------------------------------------------------------------
    def __get_atm_properties(self, h_agl):
        
        a_sound = np.interp(h_agl, self.h_agl_atm_vec, self.a_sound_vec)
        rho = np.interp(h_agl, self.h_agl_atm_vec, self.rho_vec)
        
        return a_sound, rho
    # ----------------------------------------------------------------------------
    def __compute_aed_coeff(self, mach, alpha):

        ca = self.f_ca(alpha, mach)
        cn = self.f_cn(alpha, mach)
        xcp = self.f_xcp(alpha, mach)

        return float(ca[0]), float(cn[0]), float(xcp[0])
    # ----------------------------------------------------------------------------
    def _get_aed_tables(self):
        '''Loaded in Init: Tables and Interpolation'''

        # compute wind
        self.h_agl_wind_vec = self.table_wind_model['h_AGL'].tolist()
        self.wind_shear_vec = self.table_wind_model['w_shear'].tolist()

        # atm properties
        self.h_agl_atm_vec = self.table_atmosphere['h_AGL'].tolist()
        self.a_sound_vec = self.table_atmosphere['a_m_s2'].tolist()
        self.rho_vec = self.table_atmosphere['rho_kg_m3'].tolist()
        
        # compute_aed_coeff
        aoa_vec = self.table_AED['AOA'].tolist()
        mach_vec = self.table_AED['Mach'].tolist()
        ca_vec = self.table_AED['CA'].tolist()
        cn_vec = self.table_AED['CN'].tolist()
        xcp_vec = self.table_AED['XCP_m'].tolist()
        #
        self.f_ca = interpolate.interp2d(aoa_vec, mach_vec, ca_vec, kind='linear')
        self.f_cn = interpolate.interp2d(aoa_vec, mach_vec, cn_vec, kind='linear')
        self.f_xcp = interpolate.interp2d(aoa_vec, mach_vec, xcp_vec, kind='linear')

        return
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    def _generate_state(self):
        
        # Update
        self.world.Step(1.0 / FPS, 6 * 30, 6 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        target = (self.initial_barge_coordinates[1][0] - self.initial_barge_coordinates[0][0]) / 2 + \
                 self.initial_barge_coordinates[0][0]
        state = [
            (pos.x - target) / (W),
            (pos.y - (BARGE_HEIGHT + (LEG_DOWN / PIXELTOMETER))) / (H),
            vel.x * (W) / FPS,
            vel.y * (H) / FPS,
            self.lander.angle,
            ANGULAR_VELOCITY_AMPLIFIER * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.action_history[-1][0],
            self.action_history[-1][1],
            self.action_history[-1][2] # [rad] to normalize, use: / NOZZLE_ANGLE_LIMIT
        ]

        untransformed_state = [pos.x, pos.y, vel.x, vel.y, self.lander.angle, self.lander.angularVelocity]

        return state, untransformed_state

    # ----------------------------------------------------------------------------
    def _compute_rewards(self, state, previous_state): # part_angle left if useful
        
        reward = 0
        
        shaping = - 800 * (abs(state[0]) + abs(state[1])) \
                  - 200 * (abs(state[2]) + abs(state[3])) \
                  - 1100 * abs(state[4]) - 50 * abs(state[5]) \
                  + 80 * state[6] + 80 * state[7]

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # -> cubic root to relatively penalize harder low values of control
        reward = - 3 * (np.cbrt(abs(state[8])) + np.cbrt(abs(state[9]))) - 0.3 * abs(state[10]) # penalize the use of engines
        # penalize engine transition from OFF to ON # https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html          
        reward += -(not np.heaviside(abs(previous_state[8]),0) and np.heaviside(abs(state[8]),0))*60
        reward += -(not np.heaviside(abs(previous_state[9]),0) and np.heaviside(abs(state[9]),0))*3

        return reward / 10

    ''' PROBLEM SPECIFIC - RENDERING and OBJECT CREATION'''

    # ----------------------------------------------------------------------------
    def _create_terrain(self, chunks):
        # Terrain Coordinates
        # self.helipad_x1 = W / 5
        # self.helipad_x2 = self.helipad_x1 + W / 5
        divisor_constant = 50 # 8  # Control the height of the sea
        self.helipad_y = HELIPAD_Y

        # Terrain
        # height = self.np_random.uniform(0, H / 6, size=(CHUNKS + 1,))
        height = np.random.normal(H / divisor_constant, 0.2, size=(chunks + 1,))
        chunk_x = [W / (chunks - 1) * i for i in range(chunks)]
        # self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        # self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        height[chunks // 2 - 2] = self.helipad_y
        height[chunks // 2 - 1] = self.helipad_y
        height[chunks // 2 + 0] = self.helipad_y
        height[chunks // 2 + 1] = self.helipad_y
        height[chunks // 2 + 2] = self.helipad_y

        return [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(chunks)], chunk_x  # smoothed Y

    # ----------------------------------------------------------------------------
    def _create_rocket(self, initial_coordinates=(W / 2, H * 0.95)):
        body_color = (1.0, 100.0 / 255.0 , 10.0 / 255.0) # new: bright orange
        second_color = (0, 0, 0) # black
        third_color = (1.0,0.2,0.1) # reddish

        # LANDER
        # Center of Mass is set in the middle of the polygon by default. x = 0 = middle.
        initial_x, initial_y = initial_coordinates
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / PIXELTOMETER) for x, y in LANDER_POLY]),
                density=21.0, # 12.0 -> 4.1 kg , 72 -> 24.6
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = body_color
        self.lander.color2 = second_color

        if isinstance(self.settings['Initial Force'], str):
            # ****************** THIS CODE HERE IS VERY CARTEADO
            self.lander.ApplyForceToCenter((
                self.np_random.uniform(-INITIAL_RANDOM_FORCE * 0.3, INITIAL_RANDOM_FORCE * 0.3),
                self.np_random.uniform(-1.3 * INITIAL_RANDOM_FORCE, -INITIAL_RANDOM_FORCE)
            ), True)
        else:
            self.lander.ApplyForceToCenter(self.settings['Initial Force'], True)

        # LEGS
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=18.0,
                    restitution=0.01,
                    friction = 1e9, # very high value to not allow sliding
                    categoryBits=0x0020,
                    maskBits=0x005)
            )
            leg.ground_contact = False
            leg.color1 = third_color
            leg.color2 = second_color
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(-i*0.07, 0), # (-i*0.16, 0), # localAnchorA=(-i * 0.3, 0),
                localAnchorB=(i*0.17, LEG_DOWN), # (i * 0.4, LEG_DOWN), # localAnchorB=(i * 0.5, LEG_DOWN),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = LEG_LOWEST
                rjd.upperAngle = LEG_HIGHEST
            else:
                rjd.lowerAngle = -LEG_HIGHEST # INVERTED ON PORPUSE
                rjd.upperAngle = -LEG_LOWEST
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        
        # NOZZLE
        self.nozzle = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / PIXELTOMETER) for x, y in NOZZLE_POLY]),
                density=55.0,
                friction=0.1,
                categoryBits=0x0040,
                maskBits=0x003,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.nozzle.color1 = (0, 0, 0)
        self.nozzle.color2 = (0, 0, 0)
        rjd = revoluteJointDef(
            bodyA=self.lander,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0.05),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=NOZZLE_TORQUE,
            motorSpeed=0,
            referenceAngle=0,
            lowerAngle=-(NOZZLE_ANGLE_LIMIT-1) * DEGTORAD,  # +- 15 degrees limit applied in practice
            upperAngle=(NOZZLE_ANGLE_LIMIT-1) * DEGTORAD
        )
        # The default behaviour of a revolute joint is to rotate without resistance.
        self.nozzle.joint = self.world.CreateJoint(rjd)
        
        # self.drawlist = [self.nozzle] + [self.lander] + self.legs
        self.drawlist = self.legs + [self.nozzle] + [self.lander]
        self.initial_mass = self.lander.mass
        self.remaining_fuel = INITIAL_FUEL_MASS_PERCENTAGE * self.initial_mass
        return

    # ----------------------------------------------------------------------------
    def _create_barge(self):
        # Landing Barge
        self.bargeHeight = BARGE_HEIGHT

        assert BARGE_LENGTH_X1_RATIO < BARGE_LENGTH_X2_RATIO, 'Barge Length X1 must be 0-1 and smaller than X2'

        x1 = BARGE_LENGTH_X1_RATIO*W
        x2 = BARGE_LENGTH_X2_RATIO*W
        self.landing_barge_coordinates = [(x1, 0.1), (x2, 0.1),
                                          (x2, self.bargeHeight), (x1, self.bargeHeight)]

        self.initial_barge_coordinates = self.landing_barge_coordinates
        self.minimum_barge_height = min(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        self.maximum_barge_height = max(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])

        barge_length = x2 - x1
        padRatio = 0.2
        self.landing_pad_coordinates = [x1 + barge_length * padRatio,
                                        x2 - barge_length * padRatio]

        self.landing_coordinates = self.get_landing_coordinates()

    # ----------------------------------------------------------------------------
    def _create_base_static_edges(self, CHUNKS, smooth_y, chunk_x):
        # Sky
        self.sky_polys = []
        # Ground
        self.ground_polys = []
        self.sea_polys = [[] for _ in range(SEA_CHUNKS)]

        # Main Base
        self.main_base = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self._create_static_edge(self.main_base, [p1, p2], 0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

            self.ground_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])

            for j in range(SEA_CHUNKS - 1):
                k = 1 - (j + 1) / SEA_CHUNKS
                self.sea_polys[j].append([(p1[0], p1[1] * k), (p2[0], p2[1] * k), (p2[0], 0), (p1[0], 0)])

        self._update_barge_static_edges()

    # ----------------------------------------------------------------------------
    def _update_barge_static_edges(self):
        if self.barge_base is not None:
            self.world.DestroyBody(self.barge_base)
        self.barge_base = None
        barge_edge_coordinates = [self.landing_barge_coordinates[2], self.landing_barge_coordinates[3]]
        self.barge_base = self.world.CreateStaticBody(shapes=edgeShape(vertices=barge_edge_coordinates))
        self._create_static_edge(self.barge_base, barge_edge_coordinates, friction=BARGE_FRICTION)

    # ----------------------------------------------------------------------------
    @staticmethod
    def _create_static_edge(base, vertices, friction):
        base.CreateEdgeFixture(
            vertices=vertices,
            density=0,
            friction=friction)
        return

    # ----------------------------------------------------------------------------
    def _create_particle(self, mass, x, y, ttl, radius=1):
        '''
        Used for both the Main Engine and Side Engines
        '''
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius= 2 * radius / SCALE, pos=(0, 0)),
                density=mass*3,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl  # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self._clean_particles(False)
        return p

    # ----------------------------------------------------------------------------
    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    # ----------------------------------------------------------------------------
    def _create_cloud(self, x_range, y_range, y_variance=0.1):
        self.cloud_poly = []
        numberofdiscretepoints = 8 # 3 is ugly

        initial_y = (VIEWPORT_H * np.random.uniform(y_range[0], y_range[1], 1)) / PIXELTOMETER
        initial_x = (VIEWPORT_W * np.random.uniform(x_range[0], x_range[1], 1)) / PIXELTOMETER

        y_coordinates = np.random.normal(0, y_variance, numberofdiscretepoints)
        x_step = np.linspace(initial_x, initial_x + np.random.uniform(1, 6), numberofdiscretepoints + 1)

        for i in range(0, numberofdiscretepoints):
            self.cloud_poly.append((x_step[i], initial_y + math.sin(3.14 * 2 * i / 50) * y_coordinates[i]))

        return self.cloud_poly

    # ----------------------------------------------------------------------------
    def _create_clouds(self):
        self.clouds = []
        # making 3 distinct regions of cloud generation
        num_clouds = int(np.random.uniform(2, 4, 1))
        for _ in range(num_clouds):
            self.clouds.append(self._create_cloud([2*0.2, 2*0.30], [1.5*0.65, 1.5*0.7], 1))
        num_clouds = int(np.random.uniform(4, 8, 1))  
        for _ in range(num_clouds):
            self.clouds.append(self._create_cloud([3*0.65, 3*0.85], [2.5*0.75, 2.5*0.8], 1))
        num_clouds = int(np.random.uniform(2, 8, 1))
        for _ in range(num_clouds):
            self.clouds.append(self._create_cloud([2*0.05, 2*0.25], [3*0.80, 3*0.90], 1))

    # ----------------------------------------------------------------------------
    def _decrease_mass(self, main_engine_power, side_engine_power):
        x = np.array([float(main_engine_power), float(side_engine_power)])
        # **************** -> Here, obviously need changes
        consumption_factor = 0.006 #0.011
        consumed_fuel = consumption_factor * np.sum(x * (MAIN_ENGINE_FUEL_COST, SIDE_ENGINE_FUEL_COST)) / PIXELTOMETER
        self.lander.mass -= consumed_fuel
        self.remaining_fuel -= consumed_fuel
        if self.remaining_fuel < 0: # break condition ??
            self.remaining_fuel = 0

    # ----------------------------------------------------------------------------
    @staticmethod
    def _create_labels(labels):
        labels_dict = {}
        y_spacing = 0
        for text in labels:
            labels_dict[text] = pyglet.text.Label(text, font_size=15, x=W / 2, y=H / 2,  # - y_spacing*H/10,
                                                  anchor_x='right', anchor_y='center', color=(0, 255, 0, 255))
            y_spacing += 1
        return labels_dict

    ''' RENDERING '''

    # ----------------------------------------------------------------------------
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # Viewer Creation
        if self.viewer is None:  # Initial run will enter here
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

        self._render_environment()
        self._render_lander()
        self.draw_marker(x=self.lander.worldCenter.x, y=self.lander.worldCenter.y, isBody=True)  # Center of Gravity

    # ----------------------------------------------------------------------------
    def refresh(self, mode='human', render=False):
        '''
        Used instead of _render in order to draw user defined drawings from controllers, e.g. trajectories
        '''
        # Viewer Creation
        if self.viewer is None:  # Initial run will enter here
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

        if render:
            self.render()
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # ----------------------------------------------------------------------------
    def _render_lander(self):
        
        # Rocket Lander
        
        # Lander and Particles
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False,
                                            linewidth=2).add_attr(t)
                else:
                    # Lander
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

    # ----------------------------------------------------------------------------
    def _render_clouds(self):
        for x in self.clouds:
            self.viewer.draw_polygon(x, color=(1.0, 1.0, 1.0))

    # ----------------------------------------------------------------------------
    def _update_particles(self):
        for obj in self.particles:
            obj.ttl -= 0.1
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

    # ----------------------------------------------------------------------------
    def _render_environment(self):
        
        # Sky Boundaries
        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0.83, 0.917, 1.0))

        # Landing Barge
        self.viewer.draw_polygon(self.landing_barge_coordinates, color=(0.1, 0.1, 0.1))
        for g in self.ground_polys:
            self.viewer.draw_polygon(g, color=(0, 0.5, 1.0))

        for i, s in enumerate(self.sea_polys):
            k = 1 - (i + 1) / SEA_CHUNKS
            for poly in s:
                self.viewer.draw_polygon(poly, color=(0, 0.5 * k, 1.0 * k + 0.5))

        if self.settings["Clouds"]:
            self._render_clouds()

        # Landing Flags
        for x in self.landing_pad_coordinates:
            flagy1 = self.landing_barge_coordinates[3][1]
            flagy2 = self.landing_barge_coordinates[2][1] + 25 / PIXELTOMETER

            polygon_coordinates = [(x, flagy2), (x, flagy2 - 10 / PIXELTOMETER), (x + 25 / PIXELTOMETER, flagy2 - 5 / PIXELTOMETER)]
            self.viewer.draw_polygon(polygon_coordinates, color=(1, 0, 0))
            self.viewer.draw_polyline(polygon_coordinates, color=(0, 0, 0))
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0.5, 0.5, 0.5))

    ''' CALLABLE DURING RUNTIME '''

    # ----------------------------------------------------------------------------
    def draw_marker(self, x, y, isBody=False):
        '''
        Draws '+' sign at the x and y coordinates.
        '''
        offset = 0.2
        isBodyOffset = offset * LANDER_LENGTH / 120
        if isBody:
            self.viewer.draw_polyline([(x, y - isBodyOffset), (x, y + isBodyOffset)], linewidth=2, color = (0.0, 0.0, 0.0))
            self.viewer.draw_polyline([(x - isBodyOffset, y), (x + isBodyOffset, y)], linewidth=2, color= (0.0, 0.0, 0.0))
        else:
            self.viewer.draw_polyline([(x, y - offset), (x, y + offset)], linewidth=2, color = (1.0, 0.0, 0.0))
            self.viewer.draw_polyline([(x - offset, y), (x + offset, y)], linewidth=2, color= (1.0, 0.0, 0.0))

    # ----------------------------------------------------------------------------
    def draw_polygon(self, color=(0.2, 0.2, 0.2), **kwargs):
        # path expected as (x,y)
        if self.viewer is not None:
            path = kwargs.get('path')
            if path is not None:
                self.viewer.draw_polygon(path, color=color)
            else:
                x = kwargs.get('x')
                y = kwargs.get('y')
                self.viewer.draw_polygon([(xx, yy) for xx, yy in zip(x, y)], color=color)

    # ----------------------------------------------------------------------------
    def draw_line(self, x, y, color=(0.2, 0.2, 0.2)):
        self.viewer.draw_polyline([(xx, yy) for xx, yy in zip(x, y)], linewidth=2, color=color)

    # ----------------------------------------------------------------------------
    def move_barge(self, x_movement, left_height, right_height):
        self.landing_barge_coordinates[0] = (
            self.landing_barge_coordinates[0][0] + x_movement, self.landing_barge_coordinates[0][1])
        self.landing_barge_coordinates[1] = (
            self.landing_barge_coordinates[1][0] + x_movement, self.landing_barge_coordinates[1][1])
        self.landing_barge_coordinates[2] = (
            self.landing_barge_coordinates[2][0] + x_movement, self.landing_barge_coordinates[2][1] + right_height)
        self.landing_barge_coordinates[3] = (
            self.landing_barge_coordinates[3][0] + x_movement, self.landing_barge_coordinates[3][1] + left_height)
        self.minimum_barge_height = min(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        self.maximum_barge_height = max(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        self._update_barge_static_edges()
        self.update_landing_coordinate(x_movement, x_movement)
        self.landing_coordinates = self.get_landing_coordinates()
        return self.landing_coordinates

    # ----------------------------------------------------------------------------
    def get_consumed_fuel(self):
        if self.lander is not None:
            return self.initial_mass - self.lander.mass

    # ----------------------------------------------------------------------------
    def get_landing_coordinates(self):
        x = (self.landing_barge_coordinates[1][0] - self.landing_barge_coordinates[0][0]) / 2 + \
            self.landing_barge_coordinates[0][0]
        y = abs(self.landing_barge_coordinates[2][1] - self.landing_barge_coordinates[3][1]) / 2 + \
            min(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        return [x, y]

    # ----------------------------------------------------------------------------
    def get_barge_top_edge_points(self):
        return flatten_array(self.landing_barge_coordinates[2:])

    # ----------------------------------------------------------------------------
    def get_state_with_barge_and_landing_coordinates(self, untransformed_state=False):
        if untransformed_state:
            state = self.untransformed_state
        else:
            state = self.state
        return flatten_array([state, [self.remaining_fuel,
                                      self.lander.mass],
                                      self.get_barge_top_edge_points(),
                                      self.get_landing_coordinates()])

    # ----------------------------------------------------------------------------
    def get_barge_to_ground_distance(self):
        initial_barge_coordinates = np.array(self.initial_barge_coordinates)
        current_barge_coordinates = np.array(self.landing_barge_coordinates)

        barge_height_offset = initial_barge_coordinates[:, 1] - current_barge_coordinates[:, 1]
        return np.max(barge_height_offset)

    # ----------------------------------------------------------------------------
    def update_landing_coordinate(self, left_landing_x, right_landing_x):
        self.landing_pad_coordinates[0] += left_landing_x
        self.landing_pad_coordinates[1] += right_landing_x

        x_lim_1 = self.landing_barge_coordinates[0][0]
        x_lim_2 = self.landing_barge_coordinates[1][0]

        if self.landing_pad_coordinates[0] <= x_lim_1:
            self.landing_pad_coordinates[0] = x_lim_1

        if self.landing_pad_coordinates[1] >= x_lim_2:
            self.landing_pad_coordinates[1] = x_lim_2

    # ----------------------------------------------------------------------------
    def get_action_history(self):
        return self.action_history

    # ----------------------------------------------------------------------------
    def clear_forces(self):
        self.world.ClearForces()

    # ----------------------------------------------------------------------------
    def get_nozzle_and_lander_angles(self):
        assert self.nozzle is not None, "Method called prematurely before initialization"
        return np.array([self.nozzle.angle, self.lander.angle, self.nozzle.joint.angle])

    # ----------------------------------------------------------------------------
    def evaluate_kinematics(self, actions):
         # *********************** -> WILL NEED CHANGES
        Fe, Fs, psi = actions
        theta = self.untransformed_state[THETA]
        ddot_x = (Fe * theta + Fe * psi + Fs) / MASS
        ddot_y = (Fe - Fe * theta * psi - Fs * theta - MASS * GRAVITY) / MASS
        ddot_theta = (Fe * psi * (L1 + LN) - L2 * Fs) / INERTIA
        return ddot_x, ddot_y, ddot_theta

    # ----------------------------------------------------------------------------
    def apply_random_x_disturbance(self, epsilon, left_or_right, x_force=RANDOM_DISTURBANCE_FORCE):
        if np.random.rand() < epsilon:
            if left_or_right:
                self.apply_disturbance('random', x_force, 0)
            else:
                self.apply_disturbance('random', -x_force, 0)

    # ----------------------------------------------------------------------------
    def apply_random_y_disturbance(self, epsilon, y_force=RANDOM_DISTURBANCE_FORCE):
        if np.random.rand() < epsilon:
            self.apply_disturbance('random', 0, -y_force)

    # ----------------------------------------------------------------------------
    def move_barge_randomly(self, epsilon, left_or_right, x_movement=0.05):
        if np.random.rand() < epsilon:
            if left_or_right:
                self.move_barge(x_movement=x_movement, left_height=0, right_height=0)
            else:
                self.move_barge(x_movement=-x_movement, left_height=0, right_height=0)

    # ----------------------------------------------------------------------------
    def adjust_dynamics(self, **kwargs):
        if kwargs.get('mass'):
            self.lander.mass = kwargs['mass']

        if kwargs.get('x_dot'):
            self.lander.linearVelocity.x = kwargs['x_dot']

        if kwargs.get('y_dot'):
            self.lander.linearVelocity.y = kwargs['y_dot']

        if kwargs.get('theta'):
            self.lander.angle = kwargs['theta']

        if kwargs.get('theta_dot'):
            self.lander.angularVelocity = kwargs['theta_dot']

        self.state, self.untransformed_state = self._generate_state()

    # ----------------------------------------------------------------------------
    def apply_disturbance(self, force, *args):# AWAIT OBLIVION

        if force is not None:
            if isinstance(force, str):
                x, y = args
                self.lander.ApplyForceToCenter((
                    self.np_random.uniform(x),
                    self.np_random.uniform(y)
                ), True)
            elif isinstance(force, tuple):
                self.lander.ApplyForceToCenter(force, True)
    # ----------------------------------------------------------------------------
    @staticmethod
    def compute_cost(state, untransformed_state=False, *args):
        len_state = len(state)
        cost_matrix = np.ones(len_state)
        cost_matrix[XX] = 10
        cost_matrix[X_DOT] = 5
        cost_matrix[Y_DOT] = 10
        cost_matrix[THETA] = 4
        cost_matrix[THETA_DOT] = 10

        state_target = np.zeros(len_state)
        if untransformed_state is True:
            state_target[XX] = args[XX]
            state_target[YY] = args[YY]

        ss = (state_target - abs(np.array(state)))
        return np.dot(ss, cost_matrix)


    ''' ROCKETLANDER CLASS METHODS ARE OVER '''
    ''' AUXILIARY FUNCTIONS AND COMPUTATIONS BELOW '''

# ----------------------------------------------------------------------------
def get_state_sample(samples, normal_state=True, untransformed_state=True):
    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': False,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2}
    env = RocketLander(simulation_settings)
    env.reset()
    state_samples = []
    while len(state_samples) < samples:
        f_main = np.random.uniform(0, 1)
        f_side = np.random.uniform(-1, 1)
        psi = np.random.uniform(-90 * DEGTORAD, 90 * DEGTORAD)
        action = [f_main, f_side, psi]
        s, r, done, info = env.step(action)
        if normal_state:
            state_samples.append(s)
        else:
            state_samples.append(
                env.get_state_with_barge_and_landing_coordinates(untransformed_state=untransformed_state))
        if done:
            env.reset()
    env.close()
    return state_samples

# ----------------------------------------------------------------------------
def flatten_array(the_list):
    return list(chain.from_iterable(the_list))

# ----------------------------------------------------------------------------
def compute_derivatives(state, action, sample_time=1 / FPS):
    simulation_settings = {'Side Engines': True,
                           'Clouds': False,
                           'Vectorized Nozzle': True,
                           'Graph': False,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': (0, 0)}

    eps = sample_time
    len_state = len(state)
    len_action = len(action)
    ss = np.tile(state, (len_state, 1))
    x1 = ss + np.eye(len_state) * eps
    x2 = ss - np.eye(len_state) * eps
    aa = np.tile(action, (len_state, 1))
    f1 = simulate_kinematics(x1, aa, simulation_settings)
    f2 = simulate_kinematics(x2, aa, simulation_settings)
    delta_x = f1 - f2
    delta_a = delta_x / 2 / eps  # Jacobian

    x3 = np.tile(state, (len_action, 1))
    u1 = np.tile(action, (len_action, 1)) + np.eye(len_action) * eps
    u2 = np.tile(action, (len_action, 1)) - np.eye(len_action) * eps
    f1 = simulate_kinematics(x3, u1, simulation_settings)
    f2 = simulate_kinematics(x3, u2, simulation_settings)
    delta__b = (f1 - f2) / 2 / eps
    delta__b = delta__b.T

    return delta_a, delta__b, delta_x

# ----------------------------------------------------------------------------
def simulate_kinematics(state, action, simulation_settings, render=False):
    next_state = np.zeros(state.shape)
    envs = [None for _ in range(len(state))]  # separate environment for memory management
    for i, (s, a) in enumerate(zip(state, action)):
        x, y, x_dot, y_dot, theta, theta_dot = s
        simulation_settings['Initial Coordinates'] = (x, y, 0, False)

        envs[i] = RocketLander(simulation_settings)
        if render:
            envs[i].render()
        envs[i].adjust_dynamics(y_dot=y_dot, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

        envs[i].step(a)
        if render:
            envs[i].render()
        next_state[i, :] = envs[i].untransformed_state
        envs[i].close()

    return next_state

# ----------------------------------------------------------------------------
def swap_array_values(array, indices_to_swap):
    for i, j in indices_to_swap:
        array[i], array[j] = array[j], array[i]
    return array

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



############################################

# FUNCTIONS WAITING OBLIVION
