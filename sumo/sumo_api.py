from dsm.dsm_api import DSMBase
from sumo.sumo_torch import SuMoTorch
import sumo.losses as losses
from sumo.utilities import train_sumo

import torch
import numpy as np
from tqdm import tqdm

class SuMo(DSMBase):
  """
    Model API to call for using the method
    Preprocess data to shape it to the right format and handle CUDA
  """

  def __init__(self, cuda = torch.cuda.is_available(), **params):
    self.params = params
    self.fitted = False
    self.cuda = cuda

  def _gen_torch_model(self, inputdim, optimizer):
    model = SuMoTorch(inputdim,
                      **self.params,
                      optimizer = optimizer).double()
    if self.cuda:
      model = model.cuda()
    return model

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    """
      This method is used to train an instance of the sumo model.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: np.ndarray
          A numpy array of the event/censoring times, \( t \).
      e: np.ndarray
          A numpy array of the event/censoring indicators, \( \delta \).
          \( \delta = 1 \) means the event took place.
      vsize: float
          Amount of data to set aside as the validation set.
      val_data: tuple
          A tuple of the validation dataset. If passed vsize is ignored.
      optimizer: str
          The choice of the gradient based optimization method. One of
          'Adam', 'RMSProp' or 'SGD'.
      random_state: float
          random seed that determines how the validation set is chosen.
    """
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data
    model = self._gen_torch_model(x_train.size(1), optimizer)
    model = train_sumo(model,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda == 2,
                         **args)

    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e):
    """
      This method computes the negative log likelihood of the given data.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: np.ndarray
          A numpy array of the event/censoring times, \( t \).
      e: np.ndarray
          A numpy array of the event/censoring indicators, \( \delta \).
          \( \delta = 1 \) means the event took place.

      Returns
        float: NLL
    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data

    if self.cuda == 2:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = losses.total_loss(self.torch_model, x_val, t_val, e_val)
    return loss.item()

  def predict_survival(self, x, t, risk = None):
    """
      This method computes the survival prediction of the given data at times t.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, \( x \).
      t: float or list
          A list of times at which evaluate the model.

      Returns
        np.array (len(x), len(t)) Survival prediction for each points
    """
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = torch.DoubleTensor([t_] * len(x)).to(x.device)
        outcomes = self.torch_model.predict(x, t_)
        scores.append(outcomes.detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")