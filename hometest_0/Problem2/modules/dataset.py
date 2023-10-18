import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence

class EEGDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices,:,:1440] # EEG batch
        batch_x1 = self.x[batch_indices,:,1440:] # Eye batch
        batch_y = self.y[batch_indices]
        return [batch_x, batch_x1], batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)