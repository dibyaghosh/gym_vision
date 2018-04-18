import gym
import gym_vision.envs 
import sys


env = gym.make(sys.argv[1])
env.reset()
for step in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    if step % 100 == 0:
        env.reset()