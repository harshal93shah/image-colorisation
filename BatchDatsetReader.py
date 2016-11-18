
import numpy as np
from scipy import misc
from skimage import color


class BatchDatset:
    files = []
    images = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list):
        self.files = records_list
        self.images = np.array([self._transform(filename) for filename in self.files])
        print (self.images.shape)

    def _transform(self, filename):
	image = misc.imread(filename)
	resize_image = misc.imresize(image,[128, 128])
	resize_image = color.rgb2lab(resize_image)
        return np.array(resize_image)

    def get_records(self):
        return self.images

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            self.epochs_completed += 1
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        images = self.images[start:end]
        return np.expand_dims(images[:, :, :, 0], axis=3), images

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
