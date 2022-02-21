import torch
import torch.nn as nn
import numpy as np

def total_loss(model, x, t, e, eps = 1e-8):

  # Go through network
  survival, intensity = model.forward(x, t, gradient = True)
  with torch.no_grad():
    survival.clamp_(eps)
    intensity.clamp_(eps)

  # Likelihood error
  error = torch.log(survival[e == 0]).sum()
  error += torch.log(intensity[e != 0]).sum()

  return - error / len(x)