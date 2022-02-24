import ray
import torch
import numpy as np

@ray.remote
class ParameterServer(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, algo, model, device):
        self.model = algo(state_dim, action_dim, n_latent_var, lr, betas, gamma, model, device)
        self.optimizer = torch.optim.Adam(self.model.policy.parameters(), betas=betas,  lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def returnModel(self):
        return self.model