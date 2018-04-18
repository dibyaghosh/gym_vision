import gym
from gym_vision import envs
import matplotlib.pyplot as plt

env_names = [k for k in envs.custom_envs.keys() if 'NV' not in k]

for env_name in env_names:
    env = gym.make(env_name)
    env.reset()
    for _ in range(20):
        obs,_,_,_ = env.step(env.action_space.sample())

    plt.figure()
    plt.imshow(obs)
    plt.savefig('docs/%s.png'%env_name)

for env_name in env_names:
    print("""
**%s**
![Alt text](docs/%s.png?raw=true "%s")"""%(env_name,env_name,env_name))