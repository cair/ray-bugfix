import ray
import torch
import numpy as np

@ray.remote
class ParameterServer(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, model):
        self.model = model(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=betas,  lr=lr)

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