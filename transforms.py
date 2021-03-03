import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models


class GaussianNoise():

    """Adds gaussian noise"""

    def __init__(self, mean=0., std=1e-3):
        """
        Args:
            mean (float, default 0): mean of the noise
            std (float, deafult 1e-3): standard deviation of the noise
        """
        self.std = std
        self.mean = mean
        

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip():

    """Saturates the values to a given interval."""

    def __init__(self, interval=(0., 1.)):
        """
        Args:
            interval (tuple, default (0, 1)): interval of saturation
        """
        self.interval = interval
        

    def __call__(self, tensor):
        return torch.clip(tensor, self.interval[0], self.interval[1])
    

    def __repr__(self):
        return self.__class__.__name__ + '(interval={0}, std={1})'.format(*self.interval)


class TensorSlice():

  """Slices a tensor at a given dimension"""

  def __call__(self, tensor, indices, dim=0):
    """
    Args:
        tensor (torch.tensor): tensor to be sliced
        indices (list of int): interval of slice
        dim (int, default 0): dimension of slice
    """
    if dim < 0: dim += tensor.dim()
    return tensor[(slice(None),) * dim + (indices, )]


class BestSquareCrop():

  """Crops the image in a square that maximizes the relevance of given targets."""

  def __init__(self, classes_weights_dict, *,
               min_stride=.2,
               classifier=models.resnet18(pretrained=True),
               classifier_input_size=(224, 224),
               ):
    """
    Args:
        classes_weights_dict (dict int:float): classes and respective weights for relevance criteria
        min_stride (float, default .2): minimum stride as percentage of longer image dimension
        classifier (torchvision.models, default resnet18): model to calculate classes probabilities
    """
    self.min_stride = min_stride
    self.classes_weights_dict = classes_weights_dict
    self.classifier = classifier
    self.classifier_input_size = classifier_input_size
    self.num_classes = classifier.fc.out_features
    self._calculate_weights()
    self.relevant_classes = list(self.classes_weights_dict.keys())


  def _calculate_weights(self):
    self.weights = [0] * self.num_classes
    for cls, weight in self.classes_weights_dict.items():
      self.weights[cls] = weight


  def __call__(self, tensor):

    # calculate convolution strides
    channels_height_width = tensor.shape
    height_width = list(channels_height_width[1:])
    square_size = int(min(*height_width))
    dim_convolution = height_width.index(max(height_width)) + 1
    ratio = max(*height_width) / square_size
    rest = (ratio - 1) % self.min_stride
    strides = np.arange(0., ratio - 1, self.min_stride)
    strides += np.linspace(0, 1, len(strides)) * rest

    # calculate convolution values
    relevances = []
    inputs = []
    resize = transforms.Resize(self.classifier_input_size)
    max_stride_size = channels_height_width[dim_convolution]
    tensor_slice = TensorSlice()
    for stride in strides:
      stride_size = int(stride * square_size)
      slice = range(max(0, stride_size),
                    min(stride_size + square_size, max_stride_size))
      input = tensor_slice(tensor, slice, dim=dim_convolution)
      inputs.append(input)
      input = resize(input)
      input = input.unsqueeze(0)
      logits = self.classifier(input)
      probs = F.softmax(logits, dim=1)
      probs = probs[0, self.relevant_classes]
      relevance = sum((self.classes_weights_dict[c] * p
                       for c, p in zip(self.relevant_classes, probs)))
      relevances.append(relevance)

    best_relevance = max(relevances)
    best_stride = relevances.index(best_relevance)

    return inputs[best_stride]