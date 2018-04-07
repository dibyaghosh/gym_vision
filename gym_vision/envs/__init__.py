import gym
import gym.envs.mujoco

custom_envs = {
        "TestEnv-v0": dict(
            path='gym_vision.envs.test:TestEnv',
            max_episode_steps=100,
            reward_threshold=0.0,
            kwargs=dict()
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
