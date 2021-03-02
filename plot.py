import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import product


def plot_confusion_matrix(y_true, y_pred, labels_dict=None):
  """Plots the confusion matrix.
  
  Args:
    y_true (NumPy array): Ground truth (correct) target values
    y_pred (NumPy array): Estimated targets as returned by a classifier
    labels_dict (dict, optional): Map from elements in y_true to respective label
  """
  cm = confusion_matrix(y_true, y_pred)
  
  if labels_dict is None:
    ticks = list(range(cm.shape[0]))
    labels = ticks
  else:
    ticks = range(len(labels_dict.keys()))
    labels = [labels_dict[i] for i in ticks]
    
  print(ticks)
  plt.yticks(ticks=ticks, labels=labels)
  plt.xticks(ticks=ticks, labels=labels, rotation=90)
  
  plt.gcf().patch.set_facecolor('white')
  plt.ylabel('True')
  plt.xlabel('Predicted')
  plt.title('Confusion Matrix')
  plt.imshow(cm, cmap='Blues')
  
  # annotate
  for i, j in product(ticks, ticks):
      plt.gca().text(j, i, cm[i, j],
            ha="center", va="center", color="k")