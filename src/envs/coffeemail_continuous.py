import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CoffeeMailContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # Physics parameters
        self.mass = 100.  # gram
        self.force = 100.  # Newton
        self.timestep = 1.  # second
        self.acceleration = [0, 0]  # [x, y] m/s2
        self.acceleration_decay = 0.9
        self.boundary_min = 0.
        self.boundary_max = 100.
        self.coffee_loc = [10, 20]
        self.mail_loc = [70, 90]
        self.buffer_room = 10
        self.officeA_loc = [self.boundary_min, self.boundary_max - self.buffer_room/2.]
        self.officeB_loc = [self.boundary_max - self.buffer_room/2., self.boundary_min + self.buffer_room*1.5]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.boundary_min, self.boundary_max, shape=(6,), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.episode_history = []

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        acceleration = self.acceleration
        if action == 0 or action == 1:
            acceleration[int(action)] += self.force / self.mass
        if action == 2 or action == 3:
            acceleration[int(action)-2] -= self.force / self.mass
        posx = state[0] + (self.timestep ** 2 * acceleration[0])
        posy = state[1] + (self.timestep ** 2 * acceleration[1])
        agent_has_coffee = bool(state[2])
        # agent_has_mail = bool(state[3])
        who_wants_coffee = bool(state[3])
        # B_wants_coffee = bool(state[5])

        if posx > self.boundary_max:
            posx = self.boundary_max
        if posx < self.boundary_min:
            posx = self.boundary_min
        if posy > self.boundary_max:
            posy = self.boundary_max
        if posy < self.boundary_min:
            posy = self.boundary_min

        self.episode_history.append(self.state)

        coffee_reached = (self.coffee_loc[0] - self.buffer_room < posx < self.coffee_loc[0] + self.buffer_room) \
                       and (self.coffee_loc[1] - self.buffer_room < posy < self.coffee_loc[1] + self.buffer_room)

        mail_reached = (self.mail_loc[0] - self.buffer_room < posx < self.mail_loc[0] + self.buffer_room) \
                       and (self.mail_loc[1] - self.buffer_room < posy < self.mail_loc[1] + self.buffer_room)

        officeA_reached = (self.officeA_loc[0] - self.buffer_room*2 < posx < self.officeA_loc[0] + self.buffer_room*2) \
                       and (self.officeA_loc[1] - self.buffer_room*2 < posy < self.officeA_loc[1] + self.buffer_room*2)

        officeB_reached = (self.officeB_loc[0] - self.buffer_room*2 < posx < self.officeB_loc[0] + self.buffer_room*2) \
                       and (self.officeB_loc[1] - self.buffer_room*2 < posy < self.officeB_loc[1] + self.buffer_room*2)

        border_hit = posx == self.boundary_max \
               or posy == self.boundary_max \
               or posx == self.boundary_min \
               or posy == self.boundary_min

        done = False # not(A_wants_coffee or B_wants_coffee)

        agent_has_coffee = agent_has_coffee or coffee_reached
        # agent_has_mail = agent_has_mail or mail_reached

        if border_hit and not(officeA_reached or officeB_reached):
            reward = -1  # 0
        else:
            reward = -1

        if officeA_reached and who_wants_coffee and agent_has_coffee:
            reward = 20.
            agent_has_coffee = False
            A_wants_coffee = False
            done = True

        elif officeB_reached and not(who_wants_coffee) and agent_has_coffee:
            reward = 20.
            agent_has_coffee = False
            B_wants_coffee = False
            done = True

        # A_wants_coffee = A_wants_coffee and not(officeA_reached and agent_has_coffee)
        # B_wants_coffee = B_wants_coffee and not(officeB_reached and agent_has_coffee)

        self.state = [posx, posy, agent_has_coffee, who_wants_coffee]
        # self.state = [posx, posy, agent_has_coffee, agent_has_mail, A_wants_coffee, B_wants_coffee]

        # if self.steps_beyond_done is None:
        #     self.steps_beyond_done = 0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
        #     self.steps_beyond_done += 1

        self.acceleration = [a * self.acceleration_decay for a in acceleration]

        return np.array(self.state), reward, done, {}

    def reset(self):
        agent_location = self.np_random.uniform(low=self.boundary_min, high=self.boundary_max, size=(2,))
        agent_has_coffee = 0
        # agent_has_mail = 0
        who_wants_coffee = np.random.randint(0, 2)
        # A_wants_coffee = not(B_wants_coffee)
        self.state = [agent_location[0], agent_location[1], agent_has_coffee, who_wants_coffee]
        # self.state = [agent_location[0], agent_location[1], agent_has_coffee, agent_has_mail, A_wants_coffee,
        # B_wants_coffee]
        self.steps_beyond_done = None
        self.episode_history = []
        self.acceleration = [0, 0]
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = self.boundary_max
        screen_height = self.boundary_max

        object_width = self.buffer_room
        object_height = self.buffer_room

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -object_width / 2, object_width / 2, object_height / 2, -object_height / 2
            agent = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.agenttrans = rendering.Transform()
            agent.add_attr(self.agenttrans)
            self.viewer.add_geom(agent)
            coffee = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.coffeetrans = rendering.Transform()
            coffee.add_attr(self.coffeetrans)
            coffee.set_color(.5, .1, .8)
            self.viewer.add_geom(coffee)
            mail = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.mailtrans = rendering.Transform()
            mail.add_attr(self.mailtrans)
            mail.set_color(.5, .9, .5)
            self.viewer.add_geom(mail)
            office_l, office_r, office_t, office_b = -object_width, object_width, object_height, -object_height
            officeA = rendering.FilledPolygon(
                [(office_l, office_b), (office_l, office_t), (office_r, office_t), (office_r, office_b)])
            officeB = rendering.FilledPolygon(
                [(office_l, office_b), (office_l, office_t), (office_r, office_t), (office_r, office_b)])
            self.officeAtrans = rendering.Transform()
            self.officeBtrans = rendering.Transform()
            officeA.add_attr(self.officeAtrans)
            officeB.add_attr(self.officeBtrans)
            self.viewer.add_geom(officeA)
            self.viewer.add_geom(officeB)
            # border = rendering.PolyLine([(self.buffer_room/2., self.buffer_room/2.),
            #                              (self.boundary_max + self.buffer_room/2., self.buffer_room/2.),
            #                              (self.buffer_room/2., self.buffer_room/2.),
            #                              (self.buffer_room/2., self.boundary_max + self.buffer_room/2.),
            #                              (self.buffer_room/2., self.boundary_max + self.buffer_room/2.),
            #                              (self.boundary_max + self.buffer_room/2., self.boundary_max + self.buffer_room/2.),
            #                              (self.boundary_max + self.buffer_room/2., self.boundary_max + self.buffer_room/2.),
            #                              (self.boundary_max + self.buffer_room/2., self.buffer_room/2.)], False)
            # self.bordertrans = rendering.Transform()
            # border.add_attr(self.bordertrans)
            # self.viewer.add_geom(border)

        if self.state is None: return None

        if self.viewer is not None:
            if self.state[2]:  # agent has coffee
                self.viewer.geoms[0].set_color(.5, .1, .8)
            else:
                self.viewer.geoms[0].set_color(0., 0., 0.)
            if self.state[3]:  # A wants coffee
                self.viewer.geoms[3].set_color(.5, .1, .8)
            else:
                self.viewer.geoms[3].set_color(0, 0., 0.)
            if not(self.state[3]):  # B wants coffee
                self.viewer.geoms[4].set_color(.5, .1, .8)
            else:
                self.viewer.geoms[4].set_color(0, 0., 0.)

        self.officeAtrans.set_translation(self.officeA_loc[0], self.officeA_loc[1])
        self.officeBtrans.set_translation(self.officeB_loc[0], self.officeB_loc[1])
        self.coffeetrans.set_translation(self.coffee_loc[0], self.coffee_loc[1])
        self.mailtrans.set_translation(self.mail_loc[0], self.mail_loc[1])
        self.agenttrans.set_translation(self.state[0], self.state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
