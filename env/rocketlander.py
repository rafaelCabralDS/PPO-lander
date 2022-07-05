import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import logging
import pyglet
from itertools import chain

from constants import *

# contact detector similar to Lunar Lander
class ContactDetector(contactListener):
    '''
    Creates a contact listener to check when the rocket touches down.
    '''
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def begin_contact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def end_contact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def end_contact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class RocketLander(gym.Env):
    '''
    Continuous Landing of a rocket.
    '''
    metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': FPS
    }

    def __init__(self, settings):
        self._seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, -GRAVITY))
        self.main_base = None
        self.barge_base = None
        self.CONTACT_FLAG = False

        self.minimum_barge_height = 0
        self.maximum_barge_height = 0
        self.landing_coordinates = []

        self.lander = None
        self.particles = []
        self.state = []
        self.prev_shaping = None

        if settings.get('Observation Space Size'):
            self.observation_space = spaces.Box(-np.inf, np.inf, (settings.get('Observation Space Size'),))
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, (8,))
        self.lander_tilt_angle_limit = THETA_LIMIT

        self.game_over = False

        self.settings = settings
        self.dynamicLabels = {}
        self.staticLabels = {}

        self.impulsePos = (0, 0)

        # redefined to be a spaces.Box instance
        self.action_space = spaces.Box(-np.inf, np.inf, (3,)) # Main Engine, Nozzle Angle, Left/Right Engine

        self.untransformed_state = [0] * 6 # Non-normalized state

        self.reset()

    ''' INHERITED '''

    def _seed(self, seed=None):
        self.np_random, returned_seed = seeding.np_random(seed)
        return returned_seed

    def reset(self):
        self.destroy()
        self.game_over = False
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        smoothed_terrain_edges, terrain_divider_coordinates_x = self._create_terrain(TERRAIN_CHUNKS)

        self.initial_mass = 0
        self.remaining_fuel = 0
        self.prev_shaping = 0
        self.CONTACT_FLAG = False

        # Engine Stats
        self.action_history = []

        # gradient of 0.009
        # Reference y-trajectory
        self.y_pos_ref = [1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
        self.y_pos_speed = [-1.9, -1.8, -1.64, -1.5, -1.5, -1.3, -1.0, -0.9]
        self.y_pos_flags = [False for _ in self.y_pos_ref]

        # Create the simulation objects
        self._create_clouds()
        self._create_barge()
        self._create_base_static_edges(TERRAIN_CHUNKS, smoothed_terrain_edges, terrain_divider_coordinates_x)

        # Adjust the initial coordinates of the rocket ***************
        initial_coordinates = self.settings.get('Initial Coordinates')
        if initial_coordinates is not None:
            xx, yy, randomness_degree, normalized = initial_coordinates
            x = xx * W + np.random.uniform(-randomness_degree, randomness_degree)
            y = yy * H + np.random.uniform(-randomness_degree, randomness_degree)
            if not normalized:
                x = x / W
                y = y / H
        else:
            x, y = W / 2 + np.random.uniform(-0.1, 0.1), H / self.settings['Starting Y-Pos Constant']
        self.initial_coordinates = (x, y)

        self._create_rocket(self.initial_coordinates)

        if self.settings.get('Initial State'):
            x, y, x_dot, y_dot, theta, theta_dot = self.settings.get('Initial State')
            self.adjust_dynamics(y_dot=y_dot, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

        # step through one "passive" action
        return self.step(np.array([0, 0, 0]))[0]

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


    def step(self, action):
        assert(len(action) == 3)

        # Check if contacted ground
        if (self.legs[0].ground_contact or self.legs[1].ground_contact) and self.CONTACT_FLAG == False:
            self.CONTACT_FLAG = True

        # Shutdown all Engines
        if self.CONTACT_FLAG:
            action = [0, 0, 0]

        if self.settings.get('Vectorized Nozzle'):
            part = self.nozzle
            part.angle = self.lander.angle + float(action[2])
            if part.angle > NOZZLE_ANGLE_LIMIT:
                part.angle = NOZZLE_ANGLE_LIMIT
            elif part.angle < -NOZZLE_ANGLE_LIMIT:
                part.angle = -NOZZLE_ANGLE_LIMIT

        else:
            part = self.lander 

        # "part" is used to decide where the main engine force is applied (whether it is applied to the bottom of the
        # nozzle or the bottom of the first stage rocket

        # Main FOrce Calculations
        if self.remaining_fuel == 0:
            logging.info("Strictly speaking, you're out of fuel, but act anyway.")
        m_power = self.__main_engines_force_computation(action, rocketPart=part)
        s_power, engine_dir = self.__side_engines_force_computation(action)

        if self.settings.get('Gather Stats'):
            self.action_history.append([m_power, s_power * engine_dir, part.angle])

        # Spending mass to get propulsion
        self._decrease_mass(m_power, s_power)

        # State Vector and records
        self.previous_state = self.state 
        state, self.untransformed_state = self.__generate_state()
        self.state = state 

        # Rewards for reinforcement learning
        reward = self.__compute_rewards(state, m_power, s_power, part.angle)

        state_reset_conditions = [
            self.game_over, 
            abs(state[XX]) >= 1.0, # out of x-space
            state[YY] < 0 or state[YY] > 1.3, # out of y-space
            abs(state[THETA]) > THETA_LIMIT # Rocket tilts beyond the nominal limit
        ]
        done = False
        if any(state_reset_conditions):
            done = True
            reward = -10
        if not self.lander.awake:
            done = True
            reward = +10

        self._update_particles()

        return np.array(state), reward, done, {} # {} = info (required by parent class)

    ''' PROBLEM SPECIFIC -> PHYSICS , STATES, REWARDS'''

    # ----------------------------------------------------------------------------
    def __main_engines_force_computation(self, action, rocketPart): # removed *args argument
        # Nozzle Angle Adjustment

        # For readability
        sin = math.sin(rocketPart.angle)
        cos = math.cos(rocketPart.angle)

        # Random dispersion for the particles
        dispersion = [self.np_random.uniform(-1.0, 1.0) / SCALE for _ in range(2)]

        # Main engine 
        m_power = 0
        try:
            if (action[0] > 0.0):
                # Limits
                # *******************************
                # original implementation was wrong, now clipping correctly
                m_power = np.clip(action[0], MAIN_ENGINE_LOWER, 1.0)
                assert(m_power >= MAIN_ENGINE_LOWER and m_power <= 1.0)
                ox = sin * (4 / SCALE + 2 * dispersion[0]) - cos * dispersion[1]
                oy = -cos * (4 / SCALE + 2 * dispersion[0]) - sin * dispersion[1]
                impulse_pos = (rocketPart.position[0] + ox, rocketPart.position[1] + oy)
                
                # particles as visual decoration
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power, radius=7)
                rocketParticleImpulse = (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power)
                bodyImpulse = (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power)
                point = impulse_pos
                wake = True

                # ApplyForces
                p.ApplyForce(rocketParticleImpulse, point, wake)
                rocketPart.ApplyForce(bodyImpulse, point, wake)
        except:
            print("Erro in main engine power.")

        return m_power 

    # ----------------------------------------------------------------------------
    def __side_engines_force_computation(self, action):
        # Side engines
        dispersion = [self.np_random.uniform(-1.0, 1.0) / SCALE for _ in range(2)]
        sin = math.sin(self.lander.angle) 
        cos = math.cos(self.lander.angle)
        s_power = 0.0
        y_dir = 1 # Positioning for side Thrusters
        engine_dir = 0
        # side gas thrusters enabled and action[1] > 0.5
        if (self.settings['Side Engines'] and (np.abs(action[1])) > SIDE_ENGINE_ACTIVATE):
            # Orientation engines
            engine_dir = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), SIDE_ENGINE_ACTIVATE, 1.0)
            assert(s_power >= SIDE_ENGINE_ACTIVATE and s_power <= 1.0)

            # Positioning
            constant = (LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET) / SCALE
            dx_part1 = - sin * constant
            dx_part2 = - cos * engine_dir * SIDE_ENGINE_AWAY / SCALE
            dx = dx_part1 + dx_part2
            dy = np.sqrt(np.square(constant) - np.square(dx_part1) * y_dir - sin * engine_dir * SIDE_ENGINE_AWAY / SCALE)

            # Forces -> I don't know where this 3 came from, but i'll keep it
            ox = sin * dispersion[0] - cos * (3 * dispersion[1] + engine_dir * SIDE_ENGINE_AWAY / SCALE)
            oy = -cos * dispersion[0] - sin * (3 * dispersion[1] + engine_dir * SIDE_ENGINE_AWAY / SCALE)

            # Impulse Position
            impulse_pos = (self.lander.position[0] + dx, self.lander.position[1] + dy)

            # for plotting purposes
            self.impulsePos = (self.lander.position[0] + dx, self.lander.position[1] + dy)

            try:
                p = self._create_particle(1, impulse_pos[0], impulse_pos[1], s_power, radius=3)
                p.ApplyForce((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
                self.lander.ApplyForce((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)

            except:
                logging.error("Error due to Nan in calculating y during sqrt(l^2 - x^2). "
                                "x^2 > l^2 due to approximations on the order of approximately 1e-15.")

        return s_power, engine_dir

    # ----------------------------------------------------------------------------
    def __generate_state(self):
        # Update
        self.world.Step(1.0 / FPS, 6 * 30, 6 * 30)

        pos = self.lander. position
        vel = self.lander.linearVelocity

        target = (self.initial_barge_coordinates[1][0] - self.initial_barge_coordinates[0][0]) / 2 + self.initial_barge_coordinates[0][0]
        state = [
            (pos.x - target) / (W / 2),
            (pos.y - (self.maximum_barge_height + (LEG_DOWN / SCALE))) / (W / 2) - LANDING_VERTICAL_CALIBRATION, 
            vel.x * (W / 2) / FPS,
            vel.y * (H / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]

        untransformed_state = [pos.x, pos.y, vel.x, vel.y, self.lander.angle, self.lander.angularVelocity]

        return state, untransformed_state 
    # ----------------------------------------------------------------------------
    # part_angle left here to easily include if necessary
    def __compute_rewards(self, state, main_engine_power, side_engine_power, part_angle): 
        reward = 0
        # REWARD SHAPE -> MAY NEED TO ADAPT HERE
        # ***************************
        shaping = -200 * np.sqrt(np.square(state[0]) + np.square(state[1])) -100 * np.sqrt(np.square(state[2]) + np.square(state[3]))  \
                    -1000 * abs(state[4]) - 30 * abs(state[5]) + 20 * state[6] + 20 * state[7]
    
        # penalize increase in altitude
        if state[3] > 0:
            shaping = shaping - 1

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # penalize use of engines
        reward += -main_engine_power * 0.3
        if self.settings['Side Engines']:
            reward += -side_engine_power * 0.3

        return reward / 10

    ''' PROBLEM SPECIFIC - RENDERING AND OBJECT CREATION '''

    # ----------------------------------------------------------------------------
    # Problem specific - LINKED (I do not understand what he means here)
    def _create_terrain(self, chunks): 
        # Terrain Coordinates
        divisor_constant = 8 # render sea until where? 
        self.helipad_y = H / divisor_constant

        # Terrain
        height = np.random.normal(H / divisor_constant, 0.5, size=(chunks +1,))
        chunk_x = [W / (chunks - 1) * i for i in range(chunks)]
        height[chunks // 2 - 2] = self.helipad_y
        height[chunks // 2 - 1] = self.helipad_y
        height[chunks // 2 - 0] = self.helipad_y
        height[chunks // 2 + 1] = self.helipad_y
        height[chunks // 2 + 2] = self.helipad_y

        return [0.33 * (height[i-1] + height[i] + height[i+1]) for i in range(chunks)], chunk_x # smoothed Y

    # ----------------------------------------------------------------------------
    def _create_rocket(self, initial_coordinates=(W / 2, H / 1.2)): # carteadex
        #body_color = (1, 1, 1) # original: white color
        body_color = (1, 172.0 / 255.0 , 28.0 / 255.0) # new: bright orange
        secondary_color = (0, 0, 0) # black
        third_color = (1, 1, 1) # white




        # LANDER
        initial_x, initial_y = initial_coordinates
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010, # ????
                maskBits=0x001, # collide only with ground
                restitution=0.0) 
        )
        self.lander.color1 = body_color
        self.lander.color2 = secondary_color

        if isinstance(self.settings['Initial Force'], str):
            # ORIGINAL CODE HERE WAS BAD, CHANGED
            self.lander.ApplyForceToCenter((
                self.np_random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE),
                self.np_random.uniform(-4.0 * INITIAL_RANDOM_FORCE, -3.0 * INITIAL_RANDOM_FORCE) # CONSTANTS ARE (SOMEWHAT) ARBITRARY
            ), True)
        else:
            self.lander.ApplyForceToCenter(self.settings['Initial Force'], True)

        # LEGS
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05), # carteado mas vou manter
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=5.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x005)
            )
            leg.ground_contact = False
            leg.color1 = body_color
            leg.color2 = secondary_color
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(-i * 0.3, 0), # again, another set of carteação
                localAnchorB=(i * 0.5, LEG_DOWN),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  
            )
            if i == -1:
                rjd.lowerAngle = 40 * DEGTORAD
                rjd.upperAngle = 45 * DEGTORAD
            else:
                rjd.lowerAngle = -45 * DEGTORAD
                rjd.upperAngle = -40 * DEGTORAD
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        # NOZZLE
        self.nozzle = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in NOZZLE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0040,
                maskBits=0x003, # collide only with ground
                restitution=0.0) 
        )
        self.nozzle.color1 = third_color
        self.nozzle.color2 = secondary_color
        rjd = revoluteJointDef(
            bodyA=self.lander,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0.2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=NOZZLE_TORQUE,
            motorSpeed=0,
            referenceAngle=0,
            lowerAngle=NOZZLE_ANGLE_LIMIT+1, # less carteado than the original
            upperAngle=NOZZLE_ANGLE_LIMIT-1
        )
        # rotate without resistance
        self.nozzle.joint = self.world.CreateJoint(rjd)
        # LET'S DRAW THE ROCKET
        self.drawlist = self.legs + [self.nozzle] + [self.lander]
        self.initial_mass = self.lander.mass
        self.remaining_fuel = INITIAL_FUEL_MASS_PERCENTAGE * self.initial_mass
        return

    # ----------------------------------------------------------------------------
    def _create_barge(self):
        # Landing Barge
        self.bargeHeight = self.helipad_y * (1 + 0.6)

        assert(BARGE_LENGTH_X1_RATIO < BARGE_LENGTH_X2_RATIO, 'Barge Length X1 must be 0-1 and smaller than X2')

        x1 = BARGE_LENGTH_X1_RATIO*W
        x2 = BARGE_LENGTH_X2_RATIO*W
        self.landing_barge_coordinates = [(x1,0.1), (x2,0.1), (x2,self.bargeHeight), (x1,self.bargeHeight)]
        self.initial_barge_coordinates = self.landing_barge_coordinates
        self.minimum_barge_height = min(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        self.maximum_barge_height = max(self.landing_barge_coordinates[2][1], self.landing_barge_coordinates[3][1])
        barge_length = x2 - x1
        padRatio = 0.2
        self.landing_pad_coordinates = [x1 + barge_length * padRatio, x2 - barge_length * padRatio]
        self.landing_coordinates = self.get_landing_coordinates()

    # ----------------------------------------------------------------------------
    def _create_base_static_edges(self, CHUNKS, smooth_y, chunk_x):
        # Sky
        self.sky_polys = []
        # Ground
        self.ground_polys = []
        self.sea_polys = [[] for _ in range(SEA_CHUNKS)]

        # Main Base
        self.main_base = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0),(W, 0)]))
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i+1])
            self._create_static_edge(self.main_base, [p1, p2], 0.1)
            
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], 0)])

            for j in range(SEA_CHUNKS - 1):
                k = 1 - (j+1) / SEA_CHUNKS
                self.sea_polys[j].append([(p1[0],p1[1]* k), (p2[0],p2[1] * k), (p2[0], 0), (p1[0], 0)])

        self._update_barge_static_edges()

    # ----------------------------------------------------------------------------
    def _update_barge_static_edges(self):
        if self.barge_bage is not None:
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
            friction=friction
        )
        return

    # ----------------------------------------------------------------------------
    def _create_particle(self, mass, x, y, ttl, radius=3):
        ''' Only visual effects'''
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskbits=0x001, # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p
        
    ## stopped in _clean_particles(self, all_particles):












