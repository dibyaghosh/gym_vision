from gym_vision.envs.vision_env import VisionEnv
import numpy as np
import os.path as osp

try:
    from rllab.misc import logger as logger
    rllab = True
except:
    rllab = False 
    
from gym import utils
MODEL_DIR = osp.abspath(osp.join(osp.dirname(__file__),'assets'))

class ReacherEnv(VisionEnv,utils.EzPickle):

    def __init__(self,random_goal=True,**kwargs):
        self.random = random_goal
        utils.EzPickle.__init__(self)
        VisionEnv.__init__(self,osp.join(MODEL_DIR,'reacher.xml'),2,**kwargs)
        
    def get_goal(self):
        if self.random:
            return self.np_random.uniform(low=-.2, high=.2, size=2)
        else:
            return np.array([.1,-.1])
    
    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        self.goal = self.get_goal()
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = .6
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[:] = np.array([0, 0, 0])

    def _full_state(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def _robot_state(self):
        return np.concatenate([
            self.model.data.qpos.flat[:2],
            self.model.data.qvel.flat[:2],
        ])

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist

        self.do_simulation(a, self.frame_skip)
        
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(dist=-reward_dist)

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
        logger.record_tabular('PctInGoal', 100*np.mean(np.any(dist < 0.04,axis=1)))
        logger.record_tabular('MinMinToGoalDist', np.min(dist))