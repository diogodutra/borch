import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    precision_score, recall_score, f1_score, accuracy_score,
)


class Metrics:

  """Calculates and stores metrics"""
  
  def __init__(self, scores, str_format = '{:.3f}'):
    """
    Args:
        scores (list of str): scores to be calculated and stored
        str_format (str, optional"): format to print the scores values
    """
    self.str_format = str_format
    self.score_history = {s: [] for s in scores}


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
    plt.legend()
    plt.gcf().patch.set_facecolor('white')



class MetricsClassifier(Metrics):

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
    super().__init__(scores, str_format)
    self.score_function = {
        'precision': lambda x, y: precision_score(x, y,
                                  average=average, zero_division=zero_division),
        'recall':    lambda x, y: recall_score(x, y,
                                  average=average, zero_division=zero_division),
        'F1':        lambda x, y: f1_score(x, y,
                                  average=average, zero_division=zero_division),
        'accuracy':  accuracy_score,
    }


  def plot(self, scores=None):
    """Plots scores along epochs.
    
    Args:
      scores (list of str, optional): scores to be printed, all if ignored
    """
    super().plot(scores)

    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100)
                      for x in plt.gca().get_yticks()])

    

class MetricsRegression(Metrics):

  """Calculates and stores metrics for regression"""

  def __init__(self,
               scores = ['rmse',],
               str_format = '{:.3f}',
               ):
    """
    Args:
        scores (list of str, optional): Which metrics will be calculated and stored
        str_format (str, optional"): format to print the scores values
    """
    super().__init__(scores, str_format)
    self.score_function = {
        'rmse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=True),
    }