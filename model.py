import torch
import torch.nn as nn
import torch.nn.functional as F
from DQNSelectionStrategy import EpsilonGreedyStrategy
from torch.distributions import Categorical, MultivariateNormal
import random

class DQNFCN(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self, stateDim, action_dim, n_latent_var):
        super().__init__()
        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        # self.device = device
        self.randPolicy = {"Rand":0, "Policy":0}
        self.current_step = 0
        self.num_actions = action_dim
        self.fc1 = nn.Linear(in_features=stateDim, out_features=n_latent_var).float()
        self.fc2 = nn.Linear(in_features=n_latent_var, out_features=n_latent_var).float()
        self.out = nn.Linear(in_features=n_latent_var, out_features=action_dim).float()

    def forward(self, t):
        t = t.flatten().float()
        t = self.fc1(t).float()
        t = F.relu(t).float()
        t = self.fc2(t).float()
        t = F.relu(t).float()
        t = self.out(t).float()
        return t

    def act(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            self.randPolicy["Rand"] += 1
            action = random.randrange(self.num_actions)
            output = []
            for a in range(self.num_actions):
                if a != (action-1):
                    output.append(0.0)
                else:
                    output.append(1.0)
            return torch.tensor(output).to('cpu').detach().numpy(), 0 # explore

        else:
            self.randPolicy["Policy"] += 1
            state = torch.tensor(state, dtype=torch.float32)
            state = torch.from_numpy(state)
            with torch.no_grad():
                return F.softmax(self.forward(state).to('cpu').detach().numpy()), 0 # exploit

class PPOFCN(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        # actor

        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

# @author: Viet Nguyen <nhviet1009@gmail.com> Modifed for DQN
class ConvDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, n_latent_var):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.out = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.out(x)

    def act(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            self.randPolicy["Rand"] += 1
            action = random.randrange(self.num_actions)
            output = []
            for a in range(self.num_actions):
                if a != (action - 1):
                    output.append(0.0)
                else:
                    output.append(1.0)
            return torch.tensor(output).to('cpu').detach().numpy(), 0  # explore

        else:
            self.randPolicy["Policy"] += 1
            state = torch.tensor(state, dtype=torch.float32)
            state = torch.from_numpy(state)
            with torch.no_grad():
                return F.softmax(self.forward(state).to('cpu').detach().numpy()), 0  # exploit

#@author: Viet Nguyen <nhviet1009@gmail.com>
class ConvActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, n_latent_var):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        self.cov_var = torch.full(size=(self.num_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        # self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self):
        raise NotImplementedError

    def act(self, x):
        x = self.conv_layers(x)
        action_probs = self.actor(x)
        dist = MultivariateNormal(action_probs, self.cov_mat)
        action = dist.sample()
        return action.detach().numpy(), dist.log_prob(action).detach()

    def critic(self, x):
        return F.relu(self.critic_linear(x))

    def actor(self, x):
        return F.relu(self.actor_linear(x))

    def conv_layers(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        return F.tanh(self.linear(x.view(x.size(0), -1)))

    def evaluate(self, x, action):
        x = self.conv_layers(x)
        action_probs = self.actor(x)
        dist = MultivariateNormal(action_probs, self.cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(x)

        return action_logprobs, torch.squeeze(state_value), dist_entropy