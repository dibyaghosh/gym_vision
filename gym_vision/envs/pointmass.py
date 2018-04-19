from gym_vision.envs.vision_env import VisionEnv
import numpy as np
from gym import utils

try:
    from rllab.misc import logger as logger
    rllab = True
except:
    rllab = False 
    
import os.path as osp

MODEL_DIR = osp.abspath(osp.join(osp.dirname(__file__),'assets'))

class PointmassEnv(VisionEnv, utils.EzPickle):
    def __init__(self, sparse_reward=False,**kwargs):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        VisionEnv.__init__(self,osp.join(MODEL_DIR,'twod_maze.xml'), 2, **kwargs)
        
    def step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")
        dist = np.linalg.norm(vec_dist)
        reward_dist = - dist  # particle to target
        if self.sparse_reward:
            if (-1*reward_dist) <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = False
        return ob, reward, done, dict(distance=dist)

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _full_state(self):
        particle_position = self.get_body_com("particle")
        target_position = self.get_body_com("target")
        return np.concatenate([
            particle_position,
            target_position - particle_position,
        ])
    
    def _robot_state(self):
        return np.array([0])


    def log_diagnostics(self, paths,**kwargs):
        if not rllab:
            return
        if isinstance(paths[0]['env_infos'],list):
            dist = np.array([[t['distance'] for t in traj['env_infos']] for traj in paths])
        else:
            dist = np.array([traj['env_infos']['distance'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', np.mean(dist.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(np.min(dist, axis=1)))
        logger.record_tabular('AvgFinalToGoalDist', np.mean(dist[:,-1]))
        logger.record_tabular('PctInGoal', 100*np.mean(np.any(dist < 0.1,axis=1)))
        logger.record_tabular('MinMinToGoalDist', np.min(dist))