from gym.envs.registration import register

register(
        id='gw-v0',
        entry_point='gym_gw.envs:GridWorldEnv'
        )
register(
        id='gw-extrahard-v0',
        entry_point='gym_gw.envs:GridWorldExtraHardEnv'
        )
