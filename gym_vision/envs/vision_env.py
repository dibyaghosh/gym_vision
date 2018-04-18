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


    def _get_viewer(self):
        if self.viewer is None:
            if self.from_vision:
                visible = (VISIBLE is not None)
                self.viewer = mujoco_py.MjViewer(visible=visible, init_width=self.width, init_height=self.height, go_fast=(not visible))
            else:
                self.viewer = mujoco_py.MjViewer(visible=True)

            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        
        return self.viewer
    
    def _robot_state(self):
        return np.zeros(1)
    
    def _full_state(self):
        return np.zeros(1)
    
    def _visual_state(self):
        self._get_viewer().render()
        data, width, height = self._get_viewer().get_image()
        image = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        if width != self.width or height != self.height:
            image = cv2.resize(image,(self.width,self.height))
        image = image.astype(np.float32)/255.0
        return image

    def _get_obs(self):
        if self.from_vision:
            return self._visual_state()
        return self._full_state()