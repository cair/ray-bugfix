import numpy as np
import torch

def performNActions(DQN, N, env):
    state = env.reset()
    rewardList = []

    for t in range(N):
        prevState = state
        s = numpyToTensor(state)
        action = DQN.policy_net(s)
        # action = translateAction(torch.argmax(action, dim=-1)) #Can be translated if using DQN for Continuous actions
        state, rew, done, info = env.step(action)
        rewardList.append(rew)
        if done:
            break

    return sum(rewardList)

def numpyToTensor(state):
    s = np.expand_dims(state, axis=0)
    s = np.swapaxes(s, 1, -1)
    return torch.from_numpy(s.copy())