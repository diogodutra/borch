import torch.nn as nn
import torchvision.models as models


class TransferLearning(nn.Module):

    """Wrapper for torch.nn to speed-up downloading and reusing pretrained models."""

    def __init__(self, *,
                 num_target_classes = None, 
                 pretrained_model = models.resnet18(pretrained=True),
                 freeze_parameters = False,
                 not_freeze_last_params = 3,
                 ):
        """
        Args:
            num_target_classes (int, optional): If not ommited, replaces the last layer
            pretrained_model (nn.Module, default ResNet18): Base model for Transfer Learning
            freeze_parameters (bool, default False): turn off require_gradient from first layers
            not_freeze_last_params (int, default 3): let require_gradient on from last layers
        """
        super(TransferLearning, self).__init__()

        self.model = pretrained_model

        if freeze_parameters:
          self._freeze_feature_parameters(not_freeze_last_params)
        
        # replace last layer
        if num_target_classes is not None:
          self.model.fc = nn.Linear(self.model.fc.in_features,
                                    num_target_classes)


    def _freeze_feature_parameters(self, not_freeze_last_params):
      for child in self.model.children():
        n_params = len(list(child.parameters()))
        for i, param in enumerate(child.parameters()):
          param.requires_grad = False
          if i >= n_params - not_freeze_last_params:
            break


    def forward(self, x):
        return self.model(x)