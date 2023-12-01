from gym.envs.registration import registry,register,make,spec

register(
    id="CustomFrozenLake",
    entry_point="env.custom:CustomFrozenLakeEnv",
)