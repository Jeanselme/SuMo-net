from sumo.utilities import *

from torch.autograd import grad
import torch.nn as nn
import torch

class SuMoTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], layers_surv = [100], 
               dropout = 0., optimizer = "Adam"):
    super(SuMoTorch, self).__init__()
    self.input_dim = inputdim
    self.dropout = dropout
    self.optimizer = optimizer

    self.embedding = create_representation(inputdim, layers, self.dropout) 
    self.outcome = create_representation_positive(1 + layers[-1], layers_surv + [1], self.dropout) 

  def forward(self, x, horizon, gradient = False):
    # Go through neural network
    x_embed = self.embedding(x) # Extract unconstrained NN
    time_outcome = horizon.clone().detach().requires_grad_(gradient) # Copy with independent gradient
    survival = self.outcome(torch.cat((x_embed, time_outcome.unsqueeze(1)), 1)) # Compute survival

    # Compute gradients
    intensity = grad(survival.sum(), time_outcome, create_graph = True)[0].unsqueeze(1) if gradient else None
    
    return 1 - survival, intensity

  def predict(self, x, horizon):
    return self.forward(x, horizon)[0]