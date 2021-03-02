import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import time
from datetime import timedelta
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import gc
from .utils import PrintFile


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


class Metrics():

  """Calculates and stores metrics"""

  def __init__(self,
               scores = ['precision', 'recall', 'F1', 'accuracy'],
               average = 'weighted', zero_division = 0,
               str_format = '{:.0%}',
               ):
    """
    Args:
        scores (list of str, optional): Which metrics will be calculated and stores
        average (str, default "weighter"): "average" argument for some score functions
        zero_division (int, default 0): "zero_division" argument for some score functions
        str_format (str, default "{:.0%}"): format to print the scores values
    """
    self.str_format = str_format
    self.score_history = {s: [] for s in scores}
    self.score_function = {
        'precision': lambda x, y: precision_score(x, y,
                                  average=average, zero_division=zero_division),
        'recall':    lambda x, y: recall_score(x, y,
                                  average=average, zero_division=zero_division),
        'F1':        lambda x, y: f1_score(x, y,
                                  average=average, zero_division=zero_division),
        'accuracy':  accuracy_score,
    }

  def __call__(self, y_true, y_pred, verbose=False):
    """Calculates and stores the scores results comparing ground truth from predicted targets
    
    Args:
      y_true (NumPy array): Ground truth (correct) target values
      y_pred (NumPy array): Estimated targets as returned by a classifier
    """
    for score, fc in self.score_function.items():
      self.score_history[score].append(fc(y_true, y_pred))

    if verbose: self.print()

    
  def from_epoch(self, epoch=-1, scores=None):
    """Returns dictionary with scores from a given epoch.
    
    Args:
      epoch (int, default -1): epoch related to when the scores will be printed
      scores (list of str, optional): scores to be printed, all if ignored
    """
    if scores is None: scores = list(self.score_history.keys())
    return {s: self.score_history[s][epoch] for s in scores}


  def print(self, epoch=-1, scores=None):
    """Prints on console the scores at a certain epoch.
    
    Args:
      epoch (int, default -1): epoch related to when the scores will be printed
      scores (list of str, optional): scores to be printed, all if ignored
    """
    if scores is None: scores = list(self.score_history.keys())
    longest_string = max((len(s) for s in scores))
    for score in scores:      
      print(('{}\t' + self.str_format).format(score.rjust(longest_string),
                                self.score_history[score][epoch]))


  def plot(self, scores=None):
    """Plots scores along epochs.
    
    Args:
      scores (list of str, optional): scores to be printed, all if ignored
    """
    if scores is None: scores = list(self.score_history.keys())

    for score in scores:
      plt.plot(self.score_history[score], label=score)

    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.grid()
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100)
                      for x in plt.gca().get_yticks()])
    plt.legend()
    plt.gcf().patch.set_facecolor('white')


class TrainerClassifier():

  """Trainer customized for a PyTorch classifier model."""

  def __init__(self, model,
               loss_function = nn.CrossEntropyLoss(),
               optimizer = torch.optim.Adam,
               *,
               checkpoint_parent_folder = 'drive/MyDrive/pytorch_boilerplate',
               checkpoint_model_file = 'model.pth',
               verbose = True,
               timezone='America/Toronto',
               ):
    """
    Args:
        model (nn.Module): PyTorch model, preferebly pretrained (see TransferLearning class)
        loss_function (nn.Function, default CrossEntropyLoss): Loss function
        optimizer (torch.optim, default Adam): Optimizer for backpropagation
        checkpoint_filename (str, default None): .pth file to save state dictionary of best model
        verbose (bool, default True): print validation statistics every epoch
    """
    self._init_instance_variables()
    self.model = model.to(self.device)
    self.loss_function = loss_function
    self.optimizer = optimizer(self.model.parameters())
    self.checkpoint_parent_folder = checkpoint_parent_folder
    self.checkpoint_model_file = checkpoint_model_file
    self.verbose = verbose
    self.elapsed_seconds = 0
    self.print_file = PrintFile(parent_folder=checkpoint_parent_folder,
                                timezone=timezone)


  def _init_instance_variables(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.losses_log = {key: [] for key in ['train', 'valid', 'saved']}
    self.metrics = Metrics()
    self.best_loss = np.Inf
    self.epoch = 0


  def plot_losses(self):
    """Plots the training and validation loss values across epochs"""

    plot_x = range(len(self.losses_log['train']))

    for type_data, type_plot, color, marker in [
                        ('train', plt.plot, None, None),
                        ('valid', plt.plot, None, None),
                        ('saved', plt.scatter, 'g', 'x'),
                        ]:
      if type_data in self.losses_log.keys():
        plot_y = self.losses_log[type_data]
        if len(plot_y) > 0:
          type_plot(plot_x, plot_y, label=type_data, c=color, marker=marker)


    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.gcf().patch.set_facecolor('white')


  def _str_elapsed(self):
    elapsed_time = timedelta(seconds=int(self.elapsed_seconds))
    return f"Training time (H:MM:SS): {elapsed_time}"


  def print_elapsed(self, **kwargs):
    print(self._str_elapsed(**kwargs))


  def _str_epoch(self, *, scores=None, prefix_dict={},
                     print_header=False, divider = ' '):
    if scores is None: scores = self.metrics.from_epoch()
    prefixes = list(prefix_dict.keys())

    to_print, header = '', ''

    scores_dict = self.metrics.from_epoch()
    scores = list(scores_dict.keys())

    scores_dict.update(prefix_dict)

    lengths = [len(s) + len(divider) for s in prefixes + scores]

    header += divider.join([s.rjust(lengths[i])
                            for i, s in enumerate(prefixes + scores)])

    to_print += divider.join([
                    (self.metrics.str_format if s in scores else '{}')
                    .format(scores_dict[s])
                    .rjust(lengths[i])
                    for i, s in enumerate(prefixes + scores)])

    if print_header: to_print = header + '\n' + to_print
    return to_print


  def _print_epoch(self, **kwargs):
    if self.verbose:
      print(self._str_epoch(**kwargs))


  def _str_epochs(self, epochs=None, **kwargs):
    if epochs is None: epochs = range(self.epoch)
    to_print = []
    for epoch in range(trainer.epoch):
      prefix_dict={
          'epoch': str(epoch + 1),
          'train_loss': '{:.4f}'.format(trainer.losses_log['train'][epoch]),
          'valid_loss': '{:.4f}'.format(trainer.losses_log['valid'][epoch]),
          'saved': u'\u221A' if trainer.losses_log['saved'] else ' ',
        }
      to_print.append(self._str_epoch(
          prefix_dict=prefix_dict, print_header=epoch==0, **kwargs))
      
    return '\n'.join(to_print)

      
  def print_epochs(self, **kwargs):
    print(self._str_epochs, **kwargs)


  def load_state_dict(self, state_dict):
    self.model.load_state_dict(state_dict)
    self.model.eval()


  def predict(self, loader, return_targets=False):
    """
    Calculates the classes predictions in the entire DataLoader
    Args:
        loader (DataLoader): loader containing images and targets
        return_targets (bool, default False): returns list of targets after predictions
    """

    self.model.eval()
    
    self.model.to(self.device)

    with torch.no_grad():
      all_preds = torch.tensor([])

      if return_targets: all_trues = torch.tensor([])
      
      for batch in loader:
          images, trues = batch

          logits = self.model(images.to(self.device))
          
          # get class prediction from logits
          preds = torch.max(logits, 1)[1]
          preds = preds.cpu()

          all_preds = torch.cat((all_preds, preds), dim=0)

          if return_targets: all_trues = torch.cat((all_trues, trues), dim=0)
          

      all_preds = all_preds.cpu().data.numpy().astype('i')
      if return_targets: all_trues = all_trues.cpu().data.numpy().astype('i')

      if return_targets:
        return all_preds, all_trues
      else:
        return all_preds


  def step_train(self, loader):
    loss_cumul = 0
    batches = len(loader)

    # progress bar
    progress = tqdm(enumerate(loader), desc="Loss: ",
                    total=batches, leave=False)
 
    # set model to training
    self.model.train()
    
    for i, data in progress:
        X, y = data[0].to(self.device), data[1].to(self.device)
        
        # training step for single batch
        self.model.zero_grad()
        outputs = self.model(X)
        loss = self.loss_function(outputs, y)
        loss.backward()
        self.optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        loss_cumul += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}".format(loss_cumul/(i+1)))
        
    # releasing unnecessary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loss_train = loss_cumul / batches

    return loss_train


  def step_valid(self, loader):
    loss_cumul = 0
    y_true, y_pred = [], []
    
    # set model to evaluating (testing)
    self.model.eval()
    with torch.no_grad():
      for i, data in enumerate(loader):
        X, y = data[0].to(self.device), data[1].to(self.device)

        outputs = self.model(X)

        loss_cumul += self.loss_function(outputs, y)

        # predicted classes
        predicted_classes = torch.max(outputs, 1)[1]

        y_true.extend(y.cpu())
        y_pred.extend(predicted_classes.cpu())
    
    loss_valid = float(loss_cumul / len(loader))

    return loss_valid, y_true, y_pred 


  def run(self, train_loader, valid_loader = None, *,
            max_epochs = 10, early_stop_epochs = 5,
            ):
    """
    Runs the cycle of training and validation along some epochs

    Args:
        train_loader (DataLoader): training loader with images and targets
        valid_loader (DataLoader, optional): validation loader, valid skipped if omitted
        max_epochs (int, default 10): maximum number of epochs to run the train and valid cycle
        early_stop_epochs (int, default 5): maximum number of epochs to run without improving valid loss
    """
    start_ts = time.time()
    print_header = True
    valid_loss = 0
    epochs_without_improvement = 0
    self.epoch += 1
    
    for self.epoch in range(self.epoch, self.epoch + max_epochs):
      gc.collect()

      if epochs_without_improvement >= early_stop_epochs:
        break

      else:

        # ----------------- TRAINING  --------------------
        train_loss = self.step_train(train_loader)
        self.losses_log['train'].append(train_loss)
        check_loss = train_loss

        # ----------------- VALIDATION  ----------------- 
        if valid_loader is not None:
          valid_loss, y_true, y_pred = self.step_valid(valid_loader)              
          self.metrics(y_true, y_pred)
          check_loss = valid_loss
          self.losses_log['valid'].append(valid_loss)

        # ----------------- CHECKPOINT  ----------------- 
        save_checkpoint = (check_loss < self.best_loss)
        self.losses_log['saved'].append(check_loss if save_checkpoint else np.nan)
        if not save_checkpoint:
          epochs_without_improvement += 1
        else:
          epochs_without_improvement = 0
          self.best_loss = check_loss
          self.state_dict = self.model.state_dict()
          if self.print_file.folder is not None:
            torch.save(self.state_dict,
                      self.print_file.folder + self.checkpoint_model_file)
        
        # ----------------- LOGGING  ----------------- 
        to_print = self._str_epoch(print_header=print_header,
                  prefix_dict={'epoch': str(self.epoch),
                                'train_loss': '{:.4f}'.format(train_loss),
                                'valid_loss': '{:.4f}'.format(valid_loss),
                                'saved': u'\u221A' if save_checkpoint else ' ',
                                })
        self.print_file(to_print)
        if self.verbose: print(to_print)
        print_header = False


    self.elapsed_seconds += time.time() - start_ts
    to_print = self._str_elapsed()
    self.print_file(to_print)
    if self.verbose: print(to_print)

                
    # reload best checkpoint
    self.load_state_dict(self.state_dict)
