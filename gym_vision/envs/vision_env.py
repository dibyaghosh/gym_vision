import numpy as np
from gym.envs.mujoco import mujoco_env
import os
import cv2
import mujoco_py
from gym.spaces import Box



VISIBLE = os.environ.get('VISIBLE')


class VisionEnv(mujoco_env.MujocoEnv):
    def __init__(self,xml_name,frame_skip,from_vision=True,width=224,height=224):
        self.width = width
        self.height = height
        self.from_vision = from_vision
        mujoco_env.MujocoEnv.__init__(self, xml_name, frame_skip)
    
    def _robot_state(self):
        return np.zeros(1)
    
    def _full_state(self):
        return np.zeros(1)
    
    def _visual_state(self):
        return self.sim.render(self.width,self.height,camera_name='robot') / 255

    def _get_obs(self):
        if self.from_vision:
            return self._visual_state()
        return self._full_state()