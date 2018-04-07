from gym.envs.mujoco import reacher
import mujoco_py
import numpy as np
from gym.spaces import Box


class TestEnv(reacher.ReacherEnv):
    height = 224
    width = 224

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observation_space = Box(0, 1, shape=(self.width, self.height, 3))

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(visible=False, init_width=self.width, init_height=self.height, go_fast=True)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.0
        self.viewer.cam.elevation = -90
        self.viewer.cam.lookat[:] = np.array([0, 0, 0])

    def _get_state(self):
        return super()._get_obs()

    def _get_obs(self):
        self._get_viewer().render()
        data, width, height = self._get_viewer().get_image()
        image = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        image = image.astype(np.float32)/255.0
        return image
