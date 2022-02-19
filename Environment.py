import gym

def createEnv(env):
    env = gym.make(env)
    obs = env.reset()
    return env, obs