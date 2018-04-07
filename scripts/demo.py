import gym
from gym_vision import envs
import matplotlib.pyplot as plt

env_names = ["VisualPointMass-v0","VisualReacher-v0"]
for env_name in env_names:
    env = gym.make(env_name)
    env.reset()
    for _ in range(20):
        obs,_,_,_ = env.step(env.action_space.sample())

    plt.figure()
    plt.imshow(obs)
    plt.savefig('docs/%s.png'%env_name)

