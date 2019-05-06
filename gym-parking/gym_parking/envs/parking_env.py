import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class ParkingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        super(ParkingEnv, self).__init__()
        # source point at left bottom corner
        self.window_size = np.array([600, 600])

        # parking lot cordinates
        self._lot_center = np.array([150, 100])
        lot_size = np.array([70, 120])
        self._lot_left = self._lot_center[0] - lot_size[0]//2
        self._lot_right = self._lot_center[0] + lot_size[0]//2
        self._lot_top = self._lot_center[1] + lot_size[1]//2
        self._lot_bottom = self._lot_center[1] - lot_size[1]//2

        # car info
        self._car_size = np.array([50, 100])
        self._target_lb = np.array([self._lot_center[0] - self._car_size[0]//2,\
            self._lot_center[1] - self._car_size[1]//2])
        self._target_rb = np.array([self._lot_center[0] + self._car_size[0]//2, self._target_lb[1]])
        self._target_lt = np.array([self._target_lb[0], self._lot_center[1] + self._car_size[1]//2])
        self._target_rt = np.array([self._target_rb[0], self._target_lt[1]])

        # state (car center x, car center y, car orientation)
        low_state = np.array([0., 0., 0.])
        high_state = np.array([*self.window_size, 2*np.pi])
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        # action (inner wheel angle)
        max_wheel_angle = np.pi * 40./180
        low_action = np.array([-max_wheel_angle, -2.])
        high_action = np.array([max_wheel_angle, 2.])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # ds per frame
        # self.spf = -2.

        self.viewer = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _rotate_clockwise(self, point, theta):
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        point = np.matmul(R, point)
        return point

    def _calc_vertex(self, state):
        w, h = self._car_size
        l, b, r, t = - w/2, - h/2, w/2, h/2
        self._car_lb = np.array([l, b])
        self._car_lt = np.array([l, t])
        self._car_rt = np.array([r, t])
        self._car_rb = np.array([r, b])

        # rotate and translation
        pos = state[:2]
        theta = state[2]
        self._car_lb = self._rotate_clockwise(self._car_lb, theta) + pos
        self._car_lt = self._rotate_clockwise(self._car_lt, theta) + pos
        self._car_rt = self._rotate_clockwise(self._car_rt, theta) + pos
        self._car_rb = self._rotate_clockwise(self._car_rb, theta) + pos
    
    def reset(self):
        # rand_x = self.np_random.uniform(low=350., high=450.)
        # rand_y = self.np_random.uniform(low=250., high=350.)
        # rand_theta = self.np_random.uniform(low=np.pi/3, high=2*np.pi/3)
        # self.state = np.array([rand_x, rand_y, rand_theta])
        self.init_x = 400.
        self.init_y = 400.
        self.state = np.array([self.init_x, self.init_y, np.pi/2])
        
        self._calc_vertex(self.state)

        self._last_action = np.array([0., 0.])
        self.steps = 0
        return self.state

    def _intersect(self, line_start, line_end):
        # check lot top
        if min(line_start[1], line_end[1]) > self._lot_top:
            return False
        
        # check lot left
        if min(line_start[0], line_end[0]) <= self._lot_left:
            return True

        # check lot right
        if max(line_start[0], line_end[0]) >= self._lot_right:
            return True

        # check lot bottom
        if min(line_start[1], line_end[1]) <= self._lot_bottom:
            return True

        return False

    def _check_collision(self):
        # out of bound
        xs = np.array([self._car_lb[0], self._car_rb[0], self._car_lt[0], self._car_rt[0]])
        ys = np.array([self._car_lb[1], self._car_rb[1], self._car_lt[1], self._car_rt[1]])
        if xs.min() <= 0 or xs.max() >= self.window_size[0]\
            or ys.max() >= self.window_size[1] or ys.min() <= 0:
            return True

        # collision
        if self._intersect(self._car_lt, self._car_rt):
            return True
        if self._intersect(self._car_rt, self._car_rb):
            return True
        if self._intersect(self._car_lb, self._car_rb):
            return True
        if self._intersect(self._car_lb, self._car_lt):
            return True
    
    def _calc_top_ds(self, theta, turn_left):
        h = self._car_size[1]
        R = h/np.sin(theta)
        dx = h/np.tan(theta) - R * np.cos(theta + self.dtheta)
        dy = R * np.sin(theta + self.dtheta) - h

        # forward right or backward left
        if turn_left:
            dx = -dx
        return np.array([dx, dy])

    def _calc_bottom_ds(self, theta, turn_left):
        h = self._car_size[1]
        R = h/np.tan(theta)
        dx = R - R * np.cos(self.dtheta)
        dy = R * np.sin(self.dtheta)
        
        # forward right or backward left
        if turn_left:
            dx = -dx
        return np.array([dx, dy])

    def _calc_orientation(self):
        y_vec = np.array([0., 1.])
        car_vec = self._car_lt - self._car_lb
        cos_theta = np.dot(y_vec, car_vec)/np.linalg.norm(car_vec)
        theta = np.arccos(cos_theta)
        if car_vec[0] < 0:
            theta = 2*np.pi - theta
        
        return theta

    def step(self, action):
        # calculate next position
        if action[0] < 0:
            turn_left = True
        else:
            turn_left = False
        alpha = np.abs(action[0])
        if alpha > 1e-3:
            w, h = self._car_size
            beta = np.arctan2(h, h/np.tan(alpha) + w)
            self.dtheta = action[1]*np.sin(alpha)/h
            if not turn_left:
                alpha, beta = beta, alpha
            ds_lt = self._calc_top_ds(alpha, turn_left)
            ds_rt = self._calc_top_ds(beta, turn_left)
            ds_lb = self._calc_bottom_ds(alpha, turn_left)
            ds_rb = self._calc_bottom_ds(beta, turn_left)
        else:
            ds_lt = np.array([0, action[1]])
            ds_rt = ds_lt.copy()
            ds_lb = ds_lt.copy()
            ds_rb = ds_lt.copy()

        # translate back
        theta = self.state[2]
        self._car_lt = self._rotate_clockwise(ds_lt, theta) + self._car_lt
        self._car_rt = self._rotate_clockwise(ds_rt, theta) + self._car_rt
        self._car_lb = self._rotate_clockwise(ds_lb, theta) + self._car_lb
        self._car_rb = self._rotate_clockwise(ds_rb, theta) + self._car_rb
        next_pos = (self._car_lt + self._car_rt + self._car_lb + self._car_rb)/4

        next_theta = self._calc_orientation()

        next_state = np.array([*next_pos, next_theta])
        self.state = next_state
        # self._calc_vertex(next_state)
            
        reward, done = self._get_reward(next_state, action)

        return self.state, reward, done, {}

    def _get_reward(self, state, action):
        horizontal_err = abs(self.state[0] - self._lot_center[0])/\
            abs(self.init_x - self._lot_center[0])

        vertical_err = abs(self.state[1] - self._lot_center[1])/\
            abs(self.init_y - self._lot_center[1])

        theta_err = state[2]
        if theta_err > np.pi:
            theta_err = 2*np.pi - theta_err
        theta_err /= np.pi

        reward = 1 - 0.3*horizontal_err - 0.2*vertical_err - 0.5*theta_err
        if action[1] * self._last_action[1] < 0:
            reward -= 1

        # check done
        self.steps += 1
        if self._check_collision():
            reward -= 1
            done = True
        elif state[1] <= self._lot_center[1]:
            reward += 10
            done = True
        elif self.steps >= 500:
            done = True
        else:
            done = False

        # steering_angle = np.abs(self._last_action - action[0]) * 180/np.pi
        # if steering_angle > 3.5:
        #     reward -= 0.05 * steering_angle

        self._last_action = action

        return reward, done

    def render(self, mode='human'):
        screen_width = self.window_size[0]
        screen_height = self.window_size[1]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # draw lines
            line1 = rendering.Line((0, self._lot_top), (self._lot_left, self._lot_top))
            line2 = rendering.Line((self._lot_left, self._lot_top), (self._lot_left, self._lot_bottom))
            line3 = rendering.Line((self._lot_left, self._lot_bottom), (self._lot_right, self._lot_bottom))
            line4 = rendering.Line((self._lot_right, self._lot_bottom), (self._lot_right, self._lot_top))
            line5 = rendering.Line((self._lot_right, self._lot_top), (screen_width-1, self._lot_top))
            self.viewer.add_geom(line1)
            self.viewer.add_geom(line2)
            self.viewer.add_geom(line3)
            self.viewer.add_geom(line4)
            self.viewer.add_geom(line5)

            # draw car
            l,r,t,b = -self._car_size[0]//2, self._car_size[0]//2,\
                self._car_size[1]//2, -self._car_size[1]//2
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # car.add_attr(rendering.Transform(\
            #     translation=(self.state[0], self.state[1]), rotation=self.state[2]))
            self._cartrans = rendering.Transform()
            car.add_attr(self._cartrans)
            self.viewer.add_geom(car)

            # target vertex
            target_lb = rendering.make_circle(5)
            target_rb = rendering.make_circle(5)
            target_lt = rendering.make_circle(5)
            target_rt = rendering.make_circle(5)
            target_lb.set_color(0, 0, 255)
            target_rb.set_color(0, 0, 255)
            target_lt.set_color(0, 0, 255)
            target_rt.set_color(0, 0, 255)
            target_lb.add_attr(rendering.Transform(translation=(self._target_lb[0], self._target_lb[1])))
            target_rb.add_attr(rendering.Transform(translation=(self._target_rb[0], self._target_rb[1])))
            target_lt.add_attr(rendering.Transform(translation=(self._target_lt[0], self._target_lt[1])))
            target_rt.add_attr(rendering.Transform(translation=(self._target_rt[0], self._target_rt[1])))
            self.viewer.add_geom(target_lb)
            self.viewer.add_geom(target_rb)
            self.viewer.add_geom(target_lt)
            self.viewer.add_geom(target_rt)
            
            # tracking vertex
            point_lb = rendering.make_circle(5)
            point_lt = rendering.make_circle(5)
            point_rb = rendering.make_circle(5)
            point_rt = rendering.make_circle(5)
            point_lb.set_color(255, 0, 0)
            point_lt.set_color(0, 255, 0)
            point_rb.set_color(255, 0, 0)
            point_rt.set_color(0, 255, 0)
            self._pttrans_lb = rendering.Transform()
            self._pttrans_lt = rendering.Transform()
            self._pttrans_rb = rendering.Transform()
            self._pttrans_rt = rendering.Transform()
            point_lb.add_attr(self._pttrans_lb)
            point_lt.add_attr(self._pttrans_lt)
            point_rb.add_attr(self._pttrans_rb)
            point_rt.add_attr(self._pttrans_rt)
            self.viewer.add_geom(point_lb)
            self.viewer.add_geom(point_lt)
            self.viewer.add_geom(point_rb)
            self.viewer.add_geom(point_rt)

        self._pttrans_lb.set_translation(self._car_lb[0], self._car_lb[1])
        self._pttrans_lt.set_translation(self._car_lt[0], self._car_lt[1])
        self._pttrans_rb.set_translation(self._car_rb[0], self._car_rb[1])
        self._pttrans_rt.set_translation(self._car_rt[0], self._car_rt[1])

        # move car
        self._cartrans.set_rotation(2*np.pi - self.state[2])
        self._cartrans.set_translation(int(self.state[0]), int(self.state[1]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None