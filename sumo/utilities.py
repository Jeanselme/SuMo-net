from sumo.losses import total_loss
from dsm.utilities import _reshape_tensor_with_nans

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

# Training
def get_optimizer(models, lr, optimizer, **kwargs):
	parameters = list(models.parameters())

	if optimizer == 'Adam':
		return torch.optim.Adam(parameters, lr=lr, **kwargs)
	elif optimizer == 'SGD':
		return torch.optim.SGD(parameters, lr=lr, **kwargs)
	elif optimizer == 'RMSProp':
		return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
	else:
		raise NotImplementedError('Optimizer '+optimizer+' is not implemented')

def train_sumo(model,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  n_iter = 1000, lr = 1e-3, weight_decay = 0.001,
			  bs = 100, cuda = False):
	"""	
		Train the SuMo model (should not be called directly)
	"""

	# Separate oprimizer as one might need more time to converge
	optimizer = get_optimizer(model, lr, model.optimizer, weight_decay = weight_decay)
	patience, best_loss, previous_loss = 0, np.inf, np.inf
	best_param = deepcopy(model.state_dict())
	
	nbatches = int(x_train.shape[0]/bs) + 1
	index = np.arange(len(x_train))
	t_bar = tqdm(range(n_iter))
	for i in t_bar:
		np.random.shuffle(index)
		model.train()
		for j in range(nbatches):
			xb = x_train[index[j*bs:(j+1)*bs]]
			tb = t_train[index[j*bs:(j+1)*bs]]
			eb = e_train[index[j*bs:(j+1)*bs]]
			
			if xb.shape[0] == 0:
				continue

			if cuda:
				xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

			optimizer.zero_grad()
			loss = total_loss(model,
							  xb,
							  tb,
							  eb) 
			loss.backward()
			optimizer.step()

		model.eval()
		xb, tb, eb = x_valid, t_valid, e_valid
		if cuda:
			xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
		valid_loss = total_loss(model,
								xb,
								tb,
								eb).item() 
		t_bar.set_description("Loss: {:.3f}".format(valid_loss))
		if valid_loss < previous_loss:
			patience = 0

			if valid_loss < best_loss:
				best_loss = valid_loss
				best_param = deepcopy(model.state_dict())

		elif patience == 3:
			break
		else:
			patience += 1

		previous_loss = valid_loss

	model.load_state_dict(best_param)
	return model


class PositiveLinear(nn.Module):
  """
    Neural network with positive weights
  """

  def __init__(self, in_features, out_features, bias = False):
    super(PositiveLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.log_weight)
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
      bound = np.sqrt(1 / np.sqrt(fan_in))
      nn.init.uniform_(self.bias, -bound, bound)
    self.log_weight.data.abs_().sqrt_()

  def forward(self, input):
    if self.bias is not None:
      return nn.functional.linear(input, self.log_weight ** 2, self.bias)
    else:
      return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, dropout = 0):
  """
	Create a simple multi layer neural network of positive layers
	With final Sigmoid
  """
  modules = []
  
  prevdim = inputdim
  for hidden in layers:
    modules.append(PositiveLinear(prevdim, hidden, bias=True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(nn.Tanh())
    prevdim = hidden

  # Need all values positive 
  modules[-1] = nn.Sigmoid()

  return nn.Sequential(*modules)

def create_representation(inputdim, layers, dropout = 0.5):
  modules = []
  prevdim = inputdim

  for hidden in layers:
    modules.append(nn.Linear(prevdim, hidden, bias=True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(nn.Tanh())
    prevdim = hidden
  
  return nn.Sequential(*modules)