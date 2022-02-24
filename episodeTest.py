import numpy as np
import torch

def performNActions(model, N, env):
    state = env.reset()
    rewardList = []

    for t in range(N):
        s = numpyToTensor(state)
        action, log = model.policy.act(s)
        # print(action)
        # action = translateAction(torch.argmax(action, dim=-1)) #Can be translated if using DQN for Continuous actions
        state, rew, done, info = env.step(action[0])
        rewardList.append(rew)
        if done:
            break

    return sum(rewardList)

def numpyToTensor(state):
    s = np.expand_dims(state, axis=0)
    s = np.swapaxes(s, 1, -1)
    s = s.astype(np.float32)
    return torch.from_numpy(s.copy())