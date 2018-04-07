import gym
import gym.envs.mujoco

custom_envs = {
        "VisualReacher-v0": dict(
            path='gym_vision.envs.reacher:ReacherEnv',
            max_episode_steps=100,
            reward_threshold=0.0,
            kwargs=dict()
        ),
        "VisualPointMass-v0": dict(
            path='gym_vision.envs.pointmass:VisualPointMazeEnv',
            max_episode_steps=100,
            kwargs=dict(grayscale=False,width=224,height=224)
        ),
}

def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value["path"],
                        max_episode_steps=value["max_episode_steps"],
                        kwargs=value["kwargs"])

        if "reward_threshold" in value:
            arg_dict["reward_threshold"] = value["reward_threshold"]

        gym.envs.register(**arg_dict)

register_custom_envs()
