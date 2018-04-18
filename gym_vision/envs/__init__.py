import gym
import gym.envs.mujoco

envs = [
    dict(name='Reacher',path='gym_vision.envs.reacher:ReacherEnv',length=50,random_goal=True),
    dict(name='Pointmass',path='gym_vision.envs.pointmass:PointmassEnv',length=100,sparse_reward=False),
    dict(name='FixedReacher',path='gym_vision.envs.reacher:ReacherEnv',length=50,random_goal=False),
    dict(name='SparsePointmass',path='gym_vision.envs.pointmass:PointmassEnv',length=100,sparse_reward=True)
]

custom_envs = dict()

def create_env_entries(name,path,length,**kwargs):
    
    entry_1 = dict(width=224,height=224,from_vision=True)
    entry_2 = dict(width=64,height=64,from_vision=True)
    entry_3 = dict(from_vision=False)
    
    custom_envs['VL-%s-v0'%name] = dict(path=path,max_episode_steps=length,kwargs={**kwargs, **entry_1})
    custom_envs['VS-%s-v0'%name] = dict(path=path,max_episode_steps=length,kwargs={**kwargs, **entry_2})
    custom_envs['NV-%s-v0'%name] = dict(path=path,max_episode_steps=length,kwargs={**kwargs, **entry_3})

def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value["path"],
                        max_episode_steps=value["max_episode_steps"],
                        kwargs=value["kwargs"])

        if "reward_threshold" in value:
            arg_dict["reward_threshold"] = value["reward_threshold"]

        gym.envs.register(**arg_dict)



for env in envs:
    create_env_entries(**env)
    
register_custom_envs()
