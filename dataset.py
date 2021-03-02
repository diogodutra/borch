from collections import defaultdict
from math import ceil
import copy
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Multi-Class Classification dataset."""


    def __init__(self, root_dir, transform=None, *,
                  include=None, exclude=None,
                 ):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            include (list of str, optional): only include filepaths that contain
                any of these substrings
            exclude (list of str, optional): always exclude filepaths that contain
                any of these substrings
        """
        self._init_instance_variables()
        self.root_dir = root_dir
        self.transform = transform
        self.include = include
        self.exclude = exclude
        self._list_filepaths()


    def _init_instance_variables(self):
      self.filepaths = []
      self.filepaths_organized = defaultdict(lambda: [])
      self.class_to_index = {}
      self.index_to_class = {}
      self.index_to_count = defaultdict(lambda: 0)


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, index):
        if torch.is_tensor(index): index = index.tolist()

        filepath = self.filepaths[int(index)]
        image = read_image(filepath)

        if self.transform:
            image = self.transform(image)
    
        _class = self._get_class(filepath)

        return image, self.class_to_index[_class]


    def _list_filepaths(self):
      
      classes = set()

      # crawl files in subfolders      
      for root, dirs, files in os.walk(self.root_dir):
        for file in files:
          filepath = os.path.join(root, file)
          
          add_filepath = True
          if self.include is not None:
            add_filepath = any((_inc in filepath for _inc in self.include))
          if add_filepath:
            if self.exclude is not None:
              add_filepath = all((_exc not in filepath for _exc in self.exclude))

          if add_filepath:
            #append the file name to the list of paths
            self.filepaths.append(filepath)
            
            # add country to the set of know countries
            _class = self._get_class(filepath)
            classes.add(_class)
            self.index_to_count[_class] += 1
            self.filepaths_organized[_class].append(filepath)

          
      # append the country to the targets dictionary
      for i, _class in enumerate(sorted(classes)):
          self.index_to_class[i] = _class
          self.class_to_index[_class] = i

      self.index_to_count = dict(self.index_to_count) # remove defauldict
      self.filepaths_organized = dict(self.filepaths_organized) # remove defauldict


    def _get_class(self, filepath):
      return filepath.split('/')[-2]


    def organize(self):
        class_to_filepaths = defaultdict(lambda: [])
        for filepath in self.filepaths:
            _class = self._get_class(filepath)
            class_to_filepaths[_class].append(filepath)
            
        return dict(class_to_filepaths)
    
    
    def count(self):
        class_to_filepaths = self.organize()
        class_to_count = class_to_filepaths
        for _class, filepaths in class_to_count.items():
            class_to_count[_class] = len(filepaths)
            
        return class_to_count


    def plot_occurrencies(self):
        """
        Plots the occurrencies of each label in dataset.
        """
        index_to_count = self.count()
        n_classes = len(index_to_count)
        classes = [self.index_to_class[i] for i in range(n_classes)]
        occurrencies = [index_to_count[c] for c in classes]
        plt.bar(classes, occurrencies)
        plt.gcf().patch.set_facecolor('white')
        plt.ylabel('occurrencies')
        plt.title('Dataset')
        plt.xticks(rotation=90)
        plt.gca().set_axisbelow(True)
        plt.gca().yaxis.grid(color='gray', linestyle='dashed')


    def plot_image(self, index=None):
        """
        Plots the image of a church.

        Args:
            idx (int, default random): index of the church from the dataset.filepaths list
        """
        if index is None: index = np.random.choice(range(len(self)))

        img, tgt = self[index]
        
        img = to_numpy(img)

        _ = plt.imshow(img)
        filepath = '/'.join(self.filepaths[index].split('/')[-2:])
        height, width, channels = img.shape
        title = filepath + f' ({width} x {height})'
        plt.gcf().patch.set_facecolor('white')
        plt.title(title)


    def plot_booth(self, indices=None, predictions=None, *,
                   cols=4, rows=2):
        """
        Plots many small images.

        Args:
            indices (list of int, default random repeated): indices for
                dataset.filepaths for every church to be plotted
            predictions (list of int, optional): classifier predictions for
                every index in indices, to remark hits and misses
            cols (int, default 4): columns of images
            rows (int, default 2): rows of images if indices is omitted
        """
        prediction_colors = {True: 'g', False: 'r'}
        
        if indices is None:
            idx = np.random.choice(range(len(self)))
            indices = [idx] * cols * rows # repeat the same index
        else:
            rows = ceil(len(indices) / cols)

        for i, index in enumerate(indices):
            plt.subplot(rows, cols, i+1)
            img, tgt = self[index]
            self.plot_image(index)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)

            if predictions is None:
                plt.gca().set_frame_on(False)
                plt.title(f'true:{tgt}')
            else:
                prediction = int(predictions[i])
                plt.title(f'true:{tgt} pred:{prediction}')
                correct = (prediction == tgt)
                plt.setp(plt.gca().spines.values(), linewidth=2,
                         color=prediction_colors[correct])
                

def read_image(image_path):
    """Returns NumPy array from image filename, ignoring EXIF orientation."""
    r = open(image_path,'rb').read()
    img_array = np.asarray(bytearray(r), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)[...,::-1]
    return img.copy()


def to_numpy(img):
    """
    Converts image from torch.Tensor type to NumPy array object for image plot.
    """
    if type(img) == torch.Tensor:
      img = (img * 255).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    
    return img
def stratified_split(dataset, k=1, *, shuffle=False):
    """
    Splits the dataset ensuring k samples of each country.
    
    Args:
        dataset (ChurchesDataset): dataset with all the churches images.
        k (int, default 1): occurrencies of each country

    Returns:
        dataset_split (ChurchesDataset): splitted dataset with k samples for each country
        dataset_left (ChurchesDataset): left-over from the input dataset after split
        shuffle (bool, default False): randomly selects k images from every country
    """
    dataset_split = copy.deepcopy(dataset)
    dataset_split.filepaths = []
    dataset_leftover = copy.deepcopy(dataset)
    dataset_leftover.filepaths = []

    for target, filepaths in dataset.organize().items():

        indices_all = range(len(filepaths))

        if shuffle:
            indices_split = random.sample(indices_all, k)
        else:
            indices_split = range(k)

        indices_leftover = [i for i in indices_all if i not in indices_split]

        target_filepaths_split = [filepaths[i] for i in indices_split]
        target_filepaths_leftover  = [filepaths[i] for i in indices_leftover]

        dataset_split.filepaths.extend(target_filepaths_split)
        dataset_leftover.filepaths.extend(target_filepaths_leftover)

    return dataset_split, dataset_leftover