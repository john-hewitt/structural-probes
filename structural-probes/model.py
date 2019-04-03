"""Classes for constructing word representations."""

import torch
import torch.nn as nn

class Model(nn.Module):
  '''An abstract class for neural models that
  assign a single vector to each word in a text.
  '''

  def __init__(self, args):
    super(Model, self).__init__()

  def forward(self, *args):
    """Assigns a vector to each word in a batch."""
    raise NotImplementedError("Model is an abstract class; "
        "use one of the implementing classes.")


class DiskModel(Model):
  '''A class for providing pre-computed word representations.

  Assumes the batch is constructed of loaded-from-disk
  embeddings.
  '''

  def __init__(self, args):
    super(DiskModel, self).__init__(args)

  def forward(self, batch):
    """Returns the batch itself.

    Args:
      batch: a batch of pre-computed embeddings loaded from disk.

    Returns:
      The batch, unchanged
    """
    return batch


class PyTorchModel(Model):

  def __init__(self, args, **kwargs):
    super(PyTorchModel, self).__init__(args)


class ProjectionModel(Model):
  """A class for simple contextualization of word-level embeddings.
  Runs an untrained BiLSTM on top of the loaded-from-disk embeddings.
  """

  def __init__(self, args):
    super(ProjectionModel, self).__init__(args)
    input_dim = args['model']['hidden_dim']
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=int(input_dim/2),
        num_layers=1, batch_first=True, bidirectional=True)
    for param in self.lstm.parameters():
      param.requires_grad = False
    self.lstm.to(args['device'])

  def forward(self, batch):
    """ Random BiLSTM contextualization of embeddings

    Args:
      batch: a batch of pre-computed embeddings loaded from disk.

    Returns:
      A random-init BiLSTM contextualization of the embeddings
    """
    with torch.no_grad():
      projected, _ = self.lstm(batch)
    return projected

class DecayModel(Model):
  """A class for simple contextualization of word-level embeddings.
  Computes a weighted average of the entire sentence at each word.

  """

  def __init__(self, args):
    super(DecayModel, self).__init__(args)
    self.args = args

  def forward(self, batch):
    ''' Exponential-decay contextualization of word embeddings.

    Args:
      batch: a batch of pre-computed embeddings loaded from disk.

    Returns:
      An exponentially-decaying average of the entire sequence as
      a representation for each word.
      Specifically, for word i, assigns weight:
        1 to word i
        1/2 to word (i-1,i+1)
        2/4 to word (i-2,i+2)
        ...
      before normalization by the total weight.
    '''
    forward_aggregate = torch.zeros(*batch.size(), device=self.args['device'])
    backward_aggregate = torch.zeros(*batch.size(), device=self.args['device'])
    forward_normalization_tensor = torch.zeros(batch.size()[1], device=self.args['device'])
    backward_normalization_tensor = torch.zeros(batch.size()[1], device=self.args['device'])
    batch_seq_len = torch.tensor(batch.size()[1], device=self.args['device'])
    decay_constant = torch.tensor(0.5, device=self.args['device'])
    for i in range(batch_seq_len):
      if i == 0:
        forward_aggregate[:,i,:] = batch[:,i,:]
        backward_aggregate[:,batch_seq_len-i-1,:] = batch[:,batch_seq_len-i-1,:]
        forward_normalization_tensor[i] = 1
        backward_normalization_tensor[batch_seq_len-i-1] = 1
      else:
        forward_aggregate[:,i,:] = (forward_aggregate[:,i-1,:]*decay_constant) + batch[:,i,:]
        backward_aggregate[:,batch_seq_len-i-1,:] = (backward_aggregate[:,batch_seq_len-i,:]*decay_constant) + batch[:,batch_seq_len-i-1,:]
        forward_normalization_tensor[i] = forward_normalization_tensor[i-1]*decay_constant + 1
        backward_normalization_tensor[batch_seq_len-i-1] = backward_normalization_tensor[batch_seq_len-i]*decay_constant + 1
    normalization = (forward_normalization_tensor + backward_normalization_tensor)
    normalization = normalization.unsqueeze(1).unsqueeze(0)
    decay_aggregate = (forward_aggregate + backward_aggregate) / normalization
    return decay_aggregate

