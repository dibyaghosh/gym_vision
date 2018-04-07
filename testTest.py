import gym
from gym_vision import envs

env = gym.make("TestEnv-v0")
env.reset()
for _ in range(10000):
    env.step(env.action_space.sample())
    env.render()

