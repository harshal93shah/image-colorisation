
import numpy as np
import os, sys, inspect
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

utils_path = os.path.abspath(
    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)



def read_dataset(data_dir):
    pickle_filename = "flowers_data.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)   
    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_images = result['train']      
        del result
    return training_images


