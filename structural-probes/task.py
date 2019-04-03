"""Contains classes describing linguistic tasks of interest on annotated data."""

import numpy as np
import torch

class Task:
  """Abstract class representing a linguistic task mapping texts to labels."""

  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class ParseDistanceTask(Task):
  """Maps observations to dependency parse distances between words."""

  @staticmethod
  def labels(observation):
    """Computes the distances between all pairs of words; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length, sentence_length) of distances
      in the parse tree as specified by the observation annotation.
    """
    sentence_length = len(observation[0]) #All observation fields must be of same length
    distances = torch.zeros((sentence_length, sentence_length))
    for i in range(sentence_length):
      for j in range(i,sentence_length):
        i_j_distance = ParseDistanceTask.distance_between_pairs(observation, i, j)
        distances[i][j] = i_j_distance
        distances[j][i] = i_j_distance
    return distances

  @staticmethod
  def distance_between_pairs(observation, i, j, head_indices=None):
    '''Computes path distance between a pair of words

    TODO: It would be (much) more efficient to compute all pairs' distances at once;
          this pair-by-pair method is an artefact of an older design, but
          was unit-tested for correctness... 

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: one of the two words to compute the distance between.
      j: one of the two words to compute the distance between.
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer distance d_path(i,j)
    '''
    if i == j:
      return 0
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    i_path = [i+1]
    j_path = [j+1]
    i_head = i+1
    j_head = j+1
    while True:
      if not (i_head == 0 and (i_path == [i+1] or i_path[-1] == 0)):
        i_head = head_indices[i_head - 1]
        i_path.append(i_head)
      if not (j_head == 0 and (j_path == [j+1] or j_path[-1] == 0)):
        j_head = head_indices[j_head - 1]
        j_path.append(j_head)
      if i_head in j_path:
        j_path_length = j_path.index(i_head)
        i_path_length = len(i_path) - 1
        break
      elif j_head in i_path:
        i_path_length = i_path.index(j_head)
        j_path_length = len(j_path) - 1
        break
      elif i_head == j_head:
        i_path_length = len(i_path) - 1
        j_path_length = len(j_path) - 1
        break
    total_length = j_path_length + i_path_length
    return total_length

class ParseDepthTask:
  """Maps observations to a depth in the parse tree for each word"""

  @staticmethod
  def labels(observation):
    """Computes the depth of each word; returns them as a torch tensor.

    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length,) of depths
      in the parse tree as specified by the observation annotation.
    """
    sentence_length = len(observation[0]) #All observation fields must be of same length
    depths = torch.zeros(sentence_length)
    for i in range(sentence_length):
      depths[i] = ParseDepthTask.get_ordering_index(observation, i)
    return depths

  @staticmethod
  def get_ordering_index(observation, i, head_indices=None):
    '''Computes tree depth for a single word in a sentence

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: the word in the sentence to compute the depth of
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer depth in the tree of word i
    '''
    if observation:
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices:
        if elt == '_':
          head_indices.append(0)
          number_of_underscores += 1
        else:
          head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
      i_head = head_indices[i_head - 1]
      if i_head != 0:
        length += 1
      else:
        return length

