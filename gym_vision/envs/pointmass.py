import cv2
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

from rllab.misc import logger as logger
import mujoco_py

import os.path as osp

MODEL_DIR = osp.abspath(osp.join(osp.dirname(__file__),'assets'))

class VisualPointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=1, maze_length=0.6,
                 sparse_reward=False, no_reward=False, episode_length=100, grayscale=True,
                 width=64, height=64):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length
        self.viewer = None
        self.width = width
        self.height = height
        self.grayscale=grayscale

        self.episode_length = 0

        super(VisualPointMazeEnv, self).__init__(osp.join(MODEL_DIR,'twod_maze.xml'), frame_skip=2)
        if self.grayscale:
            self.observation_space = Box(0, 1, shape=(width, height))
        else:
            self.observation_space = Box(0, 1, shape=(width, height, 3))

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(visible=False, init_width=self.width, init_height=self.height, go_fast=True)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def _step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = 0
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if (-1*reward_dist) <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        self.model.forward()
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = 90.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_state(self):
        return np.concatenate([
            self.get_body_com("particle"),
            #self.get_body_com("target"),
        ])

    def _get_obs(self):
        self._get_viewer().render()
        data, width, height = self._get_viewer().get_image()
        image = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        # rescale image to float
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)/255.0
        return image

    def log_diagnostics(self, paths,**kwargs):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        logger.record_tabular('AvgControlCost', -np.mean(rew_ctrl.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(np.min(-rew_dist, axis=1)))
        logger.record_tabular('AvgFinalToGoalDist', -np.mean(rew_dist[:,-1]))
        logger.record_tabular('PctInGoal', 100*np.mean(np.any(rew_dist > -0.1,axis=1)))
        logger.record_tabular('MinMinToGoalDist', np.min(-rew_dist))

