from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn
from Network import Network
from tflearn.optimizers import Adam 
from tflearn.layers.estimator import regression
import read_FlowersDataset as flowers
import BatchDatsetReader as dataset
import os
import scipy.io
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d, conv_2d_transpose,max_pool_2d, upsample_2d 
MAX_ITERATION = 100000
IMAGE_SIZE = 128

def main(argv=None):
    print("Reading image dataset...")
    train_images = flowers.read_dataset("./Images")
    batch_reader = dataset.BatchDatset(train_images)
    network = Network()
    adam = Adam(learning_rate=3e-3, beta1=0.9, beta2=0.99, epsilon=1.0)
    reg = regression(network, optimizer=adam)     
    l_image, color_images = batch_reader.next_batch(16)
    model = tflearn.DNN(reg, checkpoint_path='model.tfl.ckpt')
    model.fit(l_image,color_images[:,:,:,1:3])
    model.save("model.tfl")
    for itr in xrange(MAX_ITERATION):          
        model.load("model.tfl")
        l_image, color_images = batch_reader.next_batch(16)
        model = tflearn.DNN(reg, checkpoint_path='model.tfl.ckpt')
        model.fit(l_image,color_images[:,:,:,1:3], show_metric=False)
        model.save("model.tfl")
        
if __name__ == "__main__":
    tf.app.run()        

