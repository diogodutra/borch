import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from datetime import timedelta
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import gc
from .utils import PrintFile
from .metrics import MetricsRegression, MetricsClassifier

from borch.utils import PrintFile
from borch.metrics import MetricsRegression, MetricsClassifier


class Trainer:

  """Generic trainer customized for a PyTorch model."""

  def __init__(self, model,
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
    self.optimizer = optimizer(self.model.parameters())
    self.checkpoint_parent_folder = checkpoint_parent_folder
    self.checkpoint_model_file = checkpoint_model_file
    self.verbose = verbose
    self.elapsed_seconds = 0
    self.print_file = PrintFile(parent_folder=checkpoint_parent_folder,
                                timezone=timezone)


class TrainerClassifier(Trainer):

  """Trainer customized for a PyTorch classifier model."""

  def __init__(self, model,
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
    super().__init__(model, optimizer,
               checkpoint_parent_folder = checkpoint_parent_folder,
               checkpoint_model_file = checkpoint_model_file,
               verbose = verbose,
               timezone = timezone)


  def _init_instance_variables(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.losses_log = {key: [] for key in ['train', 'valid', 'saved']}
    self.metrics = MetricsClassifier()
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


  def predict(self, dataset, *,
            batch_size = 32, num_workers = 4, shuffle=False,
            return_targets=False):
    """
    Calculates the classes predictions in the entire DataLoader
    Args:
        dataset (Dataset): dataset containing images and targets
        batch_size (int, default 32): size of each batch
        num_workers (int, default 4): number of parallel threads
        return_targets (bool, default False): returns list of targets after predictions
    """

    self.model.eval()
    
    self.model.to(self.device)
    
    loader = self.create_loader(dataset,
         batch_size = batch_size, num_workers = num_workers, shuffle=shuffle)

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


  def create_loader(self, dataset, **kwargs):
    return DataLoader(dataset, **kwargs)


  def default_weights(self, dataset):
    # compensate for imbalance in dataset
    n_samples = len(dataset)
    n_classes = len(dataset.index_to_class.keys())
    classes = [dataset.index_to_class[i] for i in range(n_classes)]
    index_to_count = dataset.count()
    weights = [n_samples / index_to_count[_class] for _class in classes]
    weight_avg = sum(weights) / len(weights)
    weights = [w / weight_avg for w in weights]
    
    return weights


  def run(self, train_dataset, valid_dataset = None, *,
            loss_function = nn.CrossEntropyLoss,
            batch_size = 32, num_workers = 4, shuffle=True,
            weights = None,
            max_epochs = 10, early_stop_epochs = 5,
            ):
    """
    Runs the cycle of training and validation along some epochs

    Args:
        train_dataset (Dataset): train dataset  containing images and targets
        valid_dataset (Dataset, optional): validation dataset, valid skipped if omitted
        batch_size (int, default 32): size of each batch
        num_workers (int, default 4): number of parallel threads
        shuffle (bool, default True): random sort of train DataLoader batches
        weights (torch.tensor, optional): class weigths for loss function to compensate imbalance
        max_epochs (int, default 10): maximum number of epochs to run the train and valid cycle
        early_stop_epochs (int, default 5): maximum number of epochs to run without improving valid loss
    """
    start_ts = time.time()
    print_header = True
    valid_loss = 0
    epochs_without_improvement = 0
    self.epoch += 1
    
    train_loader = self.create_loader(train_dataset,
         batch_size = batch_size, num_workers = num_workers, shuffle=shuffle)
    
    if valid_dataset is not None:
        valid_loader = self.create_loader(valid_dataset,
             batch_size = batch_size, num_workers = num_workers, shuffle=False)
        
    self.weights = weights
    if self.weights is None: self.weights = self.default_weights(train_dataset)
    self.weights = torch.Tensor(self.weights).to(self.device)
    self.loss_function = loss_function(weight=self.weights)
    
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
        if valid_dataset is not None:
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


class TrainerRegression:

  """Trainer customized for a PyTorch regression model."""

  def __init__(self, model,
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
    self.metrics = MetricsRegression()
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


  def predict(self, dataset, *,
            batch_size = 32, num_workers = 4, shuffle=False,
            return_targets=False):
    """
    Calculates the predicted regression in the entire Dataset
    Args:
        dataset (Dataset): dataset containing images and targets
        batch_size (int, default 32): size of each batch
        num_workers (int, default 4): number of parallel threads
        return_targets (bool, default False): returns list of targets after predictions
    """

    self.model.eval()
    
    self.model.to(self.device)
    
    loader = self.create_loader(dataset,
         batch_size = batch_size, num_workers = num_workers, shuffle=shuffle)

    with torch.no_grad():
      all_preds = torch.tensor([])

      if return_targets: all_trues = torch.tensor([])
      
      for batch in loader:
          images, trues = batch

          logits = self.model(images.to(self.device))
          
          # get class prediction from logits
#           preds = torch.max(logits, 1)[1]
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
        X, y = data[0].float().to(self.device), data[1].float().to(self.device)
                
        # training step for single batch
        self.model.zero_grad()
        outputs = self.model(X).squeeze()
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
        X, y = data[0].float().to(self.device), data[1].float().to(self.device)

        outputs = self.model(X).squeeze()

        loss_cumul += self.loss_function(outputs, y)

        # predicted classes
        predicted_classes = outputs.squeeze()#torch.max(outputs, 1)[1]

        y_true.extend(y.cpu())
        y_pred.extend(predicted_classes.cpu())
    
    loss_valid = float(loss_cumul / len(loader))

    return loss_valid, y_true, y_pred 


  def create_loader(self, dataset, **kwargs):
    return DataLoader(dataset, **kwargs)


  def run(self, train_dataset, valid_dataset = None, *,
            loss_function = nn.MSELoss,
            batch_size = 32, num_workers = 4, shuffle=True,
            max_epochs = 10, early_stop_epochs = 5,
            ):
    """
    Runs the cycle of training and validation along some epochs

    Args:
        train_dataset (Dataset): train dataset  containing images and targets
        valid_dataset (Dataset, optional): validation dataset, valid skipped if omitted
        batch_size (int, default 32): size of each batch
        num_workers (int, default 4): number of parallel threads
        shuffle (bool, default True): random sort of train DataLoader batches
        max_epochs (int, default 10): maximum number of epochs to run the train and valid cycle
        early_stop_epochs (int, default 5): maximum number of epochs to run without improving valid loss
    """
    start_ts = time.time()
    print_header = True
    valid_loss = 0
    epochs_without_improvement = 0
    self.epoch += 1
    
    train_loader = self.create_loader(train_dataset,
         batch_size = batch_size, num_workers = num_workers, shuffle=shuffle)
    
    if valid_dataset is not None:
        valid_loader = self.create_loader(valid_dataset,
             batch_size = batch_size, num_workers = num_workers, shuffle=False)
        
    self.loss_function = loss_function
    
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
        if valid_dataset is not None:
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
