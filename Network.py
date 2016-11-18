import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.conv import conv_2d, conv_2d_transpose,max_pool_2d, upsample_2d 
from tflearn.layers.normalization import batch_normalization
def Network():
    network = input_data(shape=[None, 128, 128, 1], name='input')
    network = conv_2d(network, 3, 3, strides=1, activation='relu') 
    layer1= max_pool_2d(network,3)     
    network = conv_2d(layer1, 64, 3, strides=1, activation='relu')
    network = conv_2d(network, 64, 3, strides=1, activation='relu')
    layer2= max_pool_2d(network,3)  
    network = conv_2d(layer2, 128, 3, strides=1, activation='relu')
    network = conv_2d(network, 128, 3, strides=1, activation='relu')   
    network = conv_2d(network, 128, 3, strides=1, activation='relu')
    network = conv_2d(network, 128, 3, strides=2, activation='relu') 
    layer3= max_pool_2d(network,3)  
    network = conv_2d(layer3, 256, 3, strides=1, activation='relu')
    network = conv_2d(network, 256, 3, strides=1, activation='relu')   
    network = conv_2d(network, 256, 3, strides=1, activation='relu')
    network = conv_2d(network, 256, 3, strides=2, activation='relu') 
    layer4= max_pool_2d(network,3,name="layer4")  
    network = conv_2d(layer4, 512, 3, strides=1, activation='relu')
    network = conv_2d(network, 512, 3, strides=1, activation='relu')   
    network = conv_2d(network, 512, 3, strides=1, activation='relu')
    network = conv_2d(network, 512, 3, strides=2, activation='relu') 
    layer5= max_pool_2d(network,3,name="layer5")  
    network = conv_2d(layer5, 256, 1, strides=1, activation='relu')    
    network = batch_normalization(network)
    network = conv_2d_transpose (network, 256, 1, [32, 32], strides=2, activation='relu') 
    layer4 = batch_normalization(layer4)
    network= tflearn.layers.merge_ops.merge ([layer4,network], mode="elemwise_sum")
    network = conv_2d (network, 128, 3, strides=1, activation='relu') 
    network = conv_2d_transpose (network, 128, 1, [64, 64], strides=2, activation='relu') 
    layer3 = batch_normalization(layer3)
    network= tflearn.layers.merge_ops.merge ([layer3,network], mode="elemwise_sum", axis=1)
    network = conv_2d (network, 64, 3, strides=1, activation='relu') 
    network = conv_2d_transpose (network, 64, 1, [128, 128], strides=2, activation='relu') 
    layer2 = batch_normalization(layer2)
    network= tflearn.layers.merge_ops.merge ([layer2,network], mode="elemwise_sum", axis=1)
    network = conv_2d (network, 3, 3, strides=1, activation='relu') 
    layer1 = batch_normalization(layer1)
    network= tflearn.layers.merge_ops.merge ([layer1,network], mode="elemwise_sum", axis=1)
    network = conv_2d (network, 3, 3, strides=1, activation='relu') 
    network = conv_2d (network, 2, 3, strides=1, activation='softmax')
    return network
    
