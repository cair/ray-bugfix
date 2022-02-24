import torch
import numpy as np
from memory import Memory
import ray

#Create the environment for each agent
from Environment import createEnv

@ray.remote
class DataWorker(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, updateAlgo, model, device, env):
        self.Algo = updateAlgo(state_dim, action_dim, n_latent_var, lr, betas, gamma, model, device)
        self.env, self.obs = createEnv(env)#env, obs = createEnv("CartPole-v0")
        self.num_actions = action_dim
        self.action_dict = self.createActionDict()

    def compute_gradients(self, weights):
        self.Algo.set_weights(weights)
        # print("Performing Actions in environment")
        memory = self.performNActions(1000)
        # print("Done with gym")
        self.Algo.update(memory)
        return self.Algo.get_gradients()

    def performNActions(self, N):
        memory = Memory()
        state = self.env.reset()
        for t in range(N):
            # prevState = torch.from_numpy(state.copy())
            s = self.numpyToTensor(state)
            action, log = self.Algo.policy.act(s)
            # action = torch.argmax(action, dim=-1) #Used with the line below
            # action = self.getOnehot(action) #Used for discretizing continuous actions
            state, rew, done, info = self.env.step(action[0])
            # prevState = torch.unsqueeze(prevState, 0)
            stateMem = torch.from_numpy(state.copy())
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            done = torch.unsqueeze(torch.tensor(done), 0)
            memory.push(s, torch.tensor(action), stateMem, rew, done, log)
            if done:
                print(t)
                state = self.env.reset()
        return memory

    def numpyToTensor(self, state):
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        s = s.astype(np.float32)
        # s = np.float(s)
        # print(s.shape)
        return torch.from_numpy(s.copy())

    def getOnehot(self, action):
        return self.action_dict[action]

    def createActionDict(self):
        actionDict= {}
        for a in range((self.num_actions)):
            tempList = []
            for b in range((self.num_actions)):
                if a == b:
                    tempList.append(1.0)
                else:
                    tempList.append(0.0)
            actionDict.update({a: tempList})
        return actionDict