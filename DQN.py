import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class DQN():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, Transition, model, device):
        # torch.cuda.set_device(0)
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)
        self.device = device
        self.transitions = Transition

        self.policy = model(state_dim, action_dim, n_latent_var).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.target_net = model(state_dim, action_dim, n_latent_var).to(self.device)
        self.policy = self.policy.float()
        self.target_net = self.target_net.float()

        self.MseLoss = nn.MSELoss()


    def update(self, memory, BATCH_SIZE, update_timestep):
        for _ in range((update_timestep//2)//BATCH_SIZE):
            batch, idx, weight = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # # to Transition of batch-arrays.
            batch = self.transitions(*zip(*batch))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.float32)
            non_final_next_states = torch.stack([s for s in batch.next_state
                                               if s is not None], dim=0)
            state_batch = torch.stack(batch.state, dim=0)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # tempStates = state_batch.cpu().detach().numpy()
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            ones = (Counter(non_final_mask.detach().numpy()))
            reward_calc = torch.zeros(int(ones[1.0]))
            state_action_values = torch.zeros(int(ones[1.0]))
            lastInd = 0
            for x in range(len(state_batch)-1, -1, -1):
                # print(x)
                # state = torch.from_numpy(state.cpu().detach().numpy())
                # state = torch.as_tensor(state)
                # if reward_batch[x] ==
                if non_final_mask[x].detach().numpy() != 0.0:
                    stateAction = self.policy(state_batch[x]).max(0)[0]
                    state_action_values[lastInd] = (stateAction)
                    reward_calc[lastInd] = (reward_batch[x])
                    lastInd += 1
            # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            # reward_calc = torch.from_numpy(np.array(reward_calc))
            # state_action_values = torch.from_numpy(np.array(state_action_values))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            # next_state_values = torch.zeros(200)
            next_state_values = torch.zeros(int(ones[1.0]))
            for y in range(len(non_final_next_states)-1, -1, -1):
                stateAction = self.target_net(non_final_next_states[y]).max(0)[0].detach()
                next_state_values[y] = stateAction

            # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(0)[0].detach()
            # next_state_values = torch.from_numpy(np.array(next_state_values))
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_calc

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        print(self.policy.randPolicy["Rand"]/(self.policy.randPolicy["Rand"]+self.policy.randPolicy["Policy"]))

    def get_weights(self):
        return {k: v.cpu() for k, v in self.policy.state_dict().items()}

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.policy.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.policy.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)