import tensorflow as tf
import tflearn
import numpy as np
from scipy import misc
from skimage import color
from skimage import io

model=model.load("model.tfl")
img=io.imread("./test/img1.jpg")
img = misc.imresize(img,[128,128])
img=color.rgb2lab(img)
img2=[]
img2.append(img)
img=np.array(img2)
l_image=np.expand_dims(img[:,:,:,0],axis=3)		  
ab_image=model.predict(l_image)
image=tf.concat(concat_dim=3, values=[l_image, ab_image])
img=image[0,:,:,:]
img=color.lab2rgb(img.astype(np.float64))
io.imsave("/home/Ashwini/ml_project/test/test_pred.png", imgout)

